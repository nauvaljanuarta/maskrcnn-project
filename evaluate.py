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

def evaluate(model_path, data_dir='data', split='test', num_classes=7):
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
                        'category_id': int(labels[i]),
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
                        'category_id': int(labels[i]),
                        'segmentation': rle,
                        'score'      : score,
                    })

    # Evaluasi dengan pycocotools
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
    else:
        print('Tidak ada prediksi segmentation mask!')

    # Visualisasi metrik evaluasi
    if bbox_stats is not None or segm_stats is not None:
        metrics = ['Mean Prec (mAP)', 'mAP@0.50', 'mAP@0.75', 'Mean Recall', 'F1-Score']
        
        def calc_f1(p, r):
            return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
        if bbox_stats is not None:
            b_map, b_map50, b_map75, b_mar = bbox_stats[0], bbox_stats[1], bbox_stats[2], bbox_stats[8]
            bbox_vals = [b_map, b_map50, b_map75, b_mar, calc_f1(b_map, b_mar)]
        else:
            bbox_vals = [0]*5
            
        if segm_stats is not None:
            s_map, s_map50, s_map75, s_mar = segm_stats[0], segm_stats[1], segm_stats[2], segm_stats[8]
            segm_vals = [s_map, s_map50, s_map75, s_mar, calc_f1(s_map, s_mar)]
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
                            
        os.makedirs('outputs', exist_ok=True)
        out_path = os.path.join('outputs', 'evaluation_metrics.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        print(f'\n[INFO] Grafik evaluasi berhasil disimpan di: {os.path.abspath(out_path)}')

if __name__ == '__main__':
    evaluate(
        model_path='checkpoints/best_model.pth',
        data_dir='data',
        split='test'
    )
