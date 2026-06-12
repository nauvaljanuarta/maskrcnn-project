# evaluate.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import SampahDataset, SimpleTransform, collate_fn
from model import get_model, CLASS_NAMES

def compute_pr_f1(coco_eval):
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']

    precision = precision[precision > -1]
    recall = recall[recall > -1]

    mean_precision = np.mean(precision) if len(precision) > 0 else 0
    mean_recall = np.mean(recall) if len(recall) > 0 else 0

    f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)

    return mean_precision, mean_recall, f1

def evaluate(model_path, data_dir='data', split='test', num_classes=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = get_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Dataset
    annotation_file = os.path.join(data_dir, split, '_annotations.coco.json')
    dataset = SampahDataset(
        root=os.path.join(data_dir, split),
        annotation_file=annotation_file,
        transforms=SimpleTransform(train=False)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn
    )

    # Kumpulkan prediksi
    results_bbox = []
    results_segm = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Evaluating'):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target['image_id'].item()
                boxes   = output['boxes'].cpu().numpy()
                labels  = output['labels'].cpu().numpy()
                scores  = output['scores'].cpu().numpy()
                masks   = output['masks'].cpu().numpy()  # [N, 1, H, W]

                for i in range(len(boxes)):
                    score = float(scores[i])
                    if score < 0.05:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    bbox_coco = [float(x1), float(y1), float(x2-x1), float(y2-y1)]

                    results_bbox.append({
                        'image_id'   : image_id,
                        'category_id': int(labels[i]) - 1,  # remap balik: model 1,2,3 → COCO 0,1,2
                        'bbox'       : bbox_coco,
                        'score'      : score,
                    })

                    # Konversi mask → RLE untuk COCO eval
                    from pycocotools import mask as maskutil
                    binary_mask = (masks[i, 0] > 0.5).astype('uint8')
                    rle = maskutil.encode(np.asfortranarray(binary_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    results_segm.append({
                        'image_id'   : image_id,
                        'category_id': int(labels[i]) - 1,  # remap balik: model 1,2,3 → COCO 0,1,2
                        'segmentation': rle,
                        'score'      : score,
                    })

    # eval pycoco
    coco_gt = COCO(annotation_file)

    # Perbaikan anotasi segmentasi dari Roboflow yang kosong/error
    for ann in coco_gt.dataset['annotations']:
        if 'segmentation' not in ann:
            x, y, w, h = ann['bbox']
            ann['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]]
        elif isinstance(ann['segmentation'], list):
            if len(ann['segmentation']) == 0:
                x, y, w, h = ann['bbox']
                ann['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]]
            elif len(ann['segmentation']) > 0 and not isinstance(ann['segmentation'][0], (list, dict)):
                ann['segmentation'] = [ann['segmentation']]
    
    coco_gt.createIndex()

    bbox_stats = None
    segm_stats = None

    print('\n============== BBOX mAP ==============')
    if results_bbox:
        coco_dt = coco_gt.loadRes(results_bbox)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        bbox_stats = coco_eval.stats
        b_prec, b_rec, b_f1 = compute_pr_f1(coco_eval)
    else:
        print('Tidak ada prediksi bbox!')

    print('\n============== SEGMENTATION mAP ==============')
    if results_segm:
        coco_dt = coco_gt.loadRes(results_segm)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        segm_stats = coco_eval.stats
        s_prec, s_rec, s_f1 = compute_pr_f1(coco_eval)
    else:
        print('Tidak ada prediksi segmentation mask!')

    # ============== CONFUSION MATRIX (TP / FP / FN) ==============
    iou_threshold = 0.5
    score_threshold = 0.3  # Threshold confidence

    # Ambil class_ids dari COCO (biasanya 0, 1, 2)
    class_ids = sorted([c['id'] for c in coco_gt.dataset['categories']])
    num_cls = len(class_ids)

    # Buat matrix ukuran (num_cls + 1) x (num_cls + 1)
    cm = np.zeros((num_cls + 1, num_cls + 1), dtype=int)

    tp_per_class = np.zeros(num_cls, dtype=int)
    fp_per_class = np.zeros(num_cls, dtype=int)
    fn_per_class = np.zeros(num_cls, dtype=int)

    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    all_img_ids = list(coco_gt.getImgIds())

    for img_id in all_img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for r in results_bbox:
            if r['image_id'] == img_id and r['score'] >= score_threshold:
                bx = r['bbox']
                pred_boxes.append([bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]])
                pred_labels.append(r['category_id'])
                pred_scores.append(r['score'])

        if pred_scores:
            sorted_idx = np.argsort(pred_scores)[::-1]
            pred_boxes = [pred_boxes[i] for i in sorted_idx]
            pred_labels = [pred_labels[i] for i in sorted_idx]

        gt_matched = [False] * len(gt_boxes)

        for pi in range(len(pred_boxes)):
            best_iou = 0
            best_gt = -1
            for gi in range(len(gt_boxes)):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pred_boxes[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            pred_cls_idx = class_ids.index(pred_labels[pi]) if pred_labels[pi] in class_ids else -1

            if best_iou >= iou_threshold and best_gt >= 0:
                gt_matched[best_gt] = True
                gt_cls_idx = class_ids.index(gt_labels[best_gt]) if gt_labels[best_gt] in class_ids else -1

                if pred_labels[pi] == gt_labels[best_gt]:
                    # True Positive
                    if pred_cls_idx >= 0:
                        tp_per_class[pred_cls_idx] += 1
                        cm[gt_cls_idx][pred_cls_idx] += 1
                else:
                    # Misclassified
                    if pred_cls_idx >= 0 and gt_cls_idx >= 0:
                        fp_per_class[pred_cls_idx] += 1
                        fn_per_class[gt_cls_idx] += 1
                        cm[gt_cls_idx][pred_cls_idx] += 1
            else:
                # False Positive (Prediksi ada, tapi GT tidak ada)
                if pred_cls_idx >= 0:
                    fp_per_class[pred_cls_idx] += 1
                    cm[num_cls][pred_cls_idx] += 1 

        for gi in range(len(gt_boxes)):
            if not gt_matched[gi]:
                # False Negative (GT ada, tapi gagal diprediksi)
                gt_cls_idx = class_ids.index(gt_labels[gi]) if gt_labels[gi] in class_ids else -1
                if gt_cls_idx >= 0:
                    fn_per_class[gt_cls_idx] += 1
                    cm[gt_cls_idx][num_cls] += 1 

    # ============== VISUALISASI CONFUSION MATRIX ==============
    # Remap label: Karena class_ids COCO itu 0, 1, 2, tapi di CLASS_NAMES id-nya 1, 2, 3
    # Kita tambahkan 1 agar namanya terbaca benar (Trash, plastic_bag, dsb.)
    base_labels = [CLASS_NAMES.get(cid + 1, f'Class_{cid}') for cid in class_ids]
    
    # Label X (Prediksi) ditambahkan kolom False Negative
    cm_labels_x = base_labels + ['Missed (FN)']
    # Label Y (Ground Truth) ditambahkan baris False Positive
    cm_labels_y = base_labels + ['Ghost Pred (FP)']

    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
    
    ax_cm.set_title(f'Confusion Matrix\n(IoU \u2265 {iou_threshold}, Conf \u2265 {score_threshold})', fontsize=14, fontweight='bold', pad=15)
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_cm.set_xticks(np.arange(num_cls + 1))
    ax_cm.set_yticks(np.arange(num_cls + 1))
    
    ax_cm.set_xticklabels(cm_labels_x, rotation=45, ha='right', fontsize=10)
    ax_cm.set_yticklabels(cm_labels_y, fontsize=10)
    
    ax_cm.set_xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=10)
    ax_cm.set_ylabel('Ground Truth (Actual)', fontsize=12, fontweight='bold', labelpad=10)

    # Tulis angka di dalam kotak
    thresh = cm.max() / 2.
    for i in range(num_cls + 1):
        for j in range(num_cls + 1):
            val = cm[i, j]
            # Sembunyikan angka 0 di cell Background-Background karena tidak relevan
            if i == num_cls and j == num_cls:
                continue 
            color = 'white' if val > thresh else 'black'
            ax_cm.text(j, i, str(val), ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    os.makedirs('outputs', exist_ok=True)
    cm_path = os.path.join('outputs', 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    print(f'[INFO] Confusion matrix disimpan di: {os.path.abspath(cm_path)}')

    # Visualisasi metrik evaluasi (grafik batang)
    if bbox_stats is not None or segm_stats is not None:
        metrics = ['Mean Prec (mAP)', 'mAP@0.50', 'mAP@0.75', 'Mean Recall', 'F1-Score']
        
        if bbox_stats is not None:
            b_map, b_map50, b_map75, b_mar = bbox_stats[0], bbox_stats[1], bbox_stats[2], bbox_stats[8]
            bbox_vals = [b_map, b_map50, b_map75, b_rec, b_f1]
        else:
            bbox_vals = [0]*5
            
        if segm_stats is not None:
            s_map, s_map50, s_map75, s_mar = segm_stats[0], segm_stats[1], segm_stats[2], segm_stats[8]
            segm_vals = [s_map, s_map50, s_map75, s_rec, s_f1]
        else:
            segm_vals = [0]*5
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, bbox_vals, width, label='BBOX', color='#1f77b4')
        rects2 = ax.bar(x + width/2, segm_vals, width, label='SEGM', color='#d62728')
        
        ax.set_ylabel('Skor (0 - 1.0)')
        ax.set_title('Metrik Evaluasi: Bounding Box vs Segmentation')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        # Tambahkan nilai di atas grafik batang
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                            
        out_path = os.path.join('outputs', 'evaluation_metrics.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        print(f'[INFO] Grafik evaluasi berhasil disimpan di: {os.path.abspath(out_path)}')

if __name__ == '__main__':
    evaluate(
        model_path='checkpoints/best_model.pth',
        data_dir='data',
        split='test'
    )