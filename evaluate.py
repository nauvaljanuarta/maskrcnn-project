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
from model import get_model

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
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    from model import CLASS_NAMES
    iou_threshold = 0.5
    score_threshold = 0.3  # Threshold untuk confusion matrix (prediksi "yakin")

    class_ids = sorted([c['id'] for c in coco_gt.dataset['categories']])
    num_cls = len(class_ids)

    # Confusion matrix: baris = GT class, kolom = Predicted class
    # Tambah 1 kolom/baris untuk "Background" (missed / false)
    cm = np.zeros((num_cls + 1, num_cls + 1), dtype=int)
    # Indeks 0..num_cls-1 = kelas 1..6, indeks num_cls = Background/Missed

    tp_per_class = np.zeros(num_cls, dtype=int)
    fp_per_class = np.zeros(num_cls, dtype=int)
    fn_per_class = np.zeros(num_cls, dtype=int)

    def compute_iou(box1, box2):
        """Hitung IoU antara 2 bbox [x1,y1,x2,y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    # Ambil semua image_id dari ground truth
    all_img_ids = list(coco_gt.getImgIds())

    for img_id in all_img_ids:
        # Ground truth boxes
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        # Predicted boxes (filter by score threshold)
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for r in results_bbox:
            if r['image_id'] == img_id and r['score'] >= score_threshold:
                bx = r['bbox']
                pred_boxes.append([bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]])
                pred_labels.append(r['category_id'])
                pred_scores.append(r['score'])

        # Sortir prediksi berdasarkan skor (tertinggi dulu)
        if pred_scores:
            sorted_idx = np.argsort(pred_scores)[::-1]
            pred_boxes = [pred_boxes[i] for i in sorted_idx]
            pred_labels = [pred_labels[i] for i in sorted_idx]

        gt_matched = [False] * len(gt_boxes)

        # Match predictions ke ground truth
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
                    # TP: Benar kelas, benar lokasi
                    if pred_cls_idx >= 0:
                        tp_per_class[pred_cls_idx] += 1
                        cm[gt_cls_idx][pred_cls_idx] += 1
                else:
                    # Salah kelas (tapi lokasi benar)
                    if pred_cls_idx >= 0 and gt_cls_idx >= 0:
                        fp_per_class[pred_cls_idx] += 1
                        fn_per_class[gt_cls_idx] += 1
                        cm[gt_cls_idx][pred_cls_idx] += 1
            else:
                # FP: Prediksi tidak cocok dengan GT manapun
                if pred_cls_idx >= 0:
                    fp_per_class[pred_cls_idx] += 1
                    cm[num_cls][pred_cls_idx] += 1  # Background -> Predicted class

        # FN: GT yang tidak terdeteksi
        for gi in range(len(gt_boxes)):
            if not gt_matched[gi]:
                gt_cls_idx = class_ids.index(gt_labels[gi]) if gt_labels[gi] in class_ids else -1
                if gt_cls_idx >= 0:
                    fn_per_class[gt_cls_idx] += 1
                    cm[gt_cls_idx][num_cls] += 1  # GT class -> Background (missed)

    # Hitung total TP/FP/FN
    total_tp = int(tp_per_class.sum())
    total_fp = int(fp_per_class.sum())
    total_fn = int(fn_per_class.sum())
    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0

    # Visualisasi Confusion Matrix (Heatmap)
    cm_labels = [CLASS_NAMES.get(cid, f'cls_{cid}') for cid in class_ids] + ['Background']

    fig_cm, ax_cm = plt.subplots(figsize=(9, 7))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
    ax_cm.set_title('Confusion Matrix (IoU >= 0.5)', fontsize=14, fontweight='bold')
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_cm.set_xticks(np.arange(num_cls + 1))
    ax_cm.set_yticks(np.arange(num_cls + 1))
    ax_cm.set_xticklabels(cm_labels, rotation=45, ha='right', fontsize=9)
    ax_cm.set_yticklabels(cm_labels, fontsize=9)
    ax_cm.set_xlabel('Predicted', fontsize=12)
    ax_cm.set_ylabel('Ground Truth', fontsize=12)

    # Tulis angka di dalam kotak
    for i in range(num_cls + 1):
        for j in range(num_cls + 1):
            val = cm[i, j]
            color = 'white' if val > cm.max() / 2 else 'black'
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
        
        def calc_f1(p, r):
            return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
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
