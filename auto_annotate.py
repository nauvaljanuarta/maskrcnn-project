# auto_annotate.py
# Script untuk semi-supervised labeling:
# Menggunakan best_model.pth untuk memprediksi anotasi pada gambar yang belum berlabel,
# lalu menyimpannya dalam format COCO JSON untuk di-review di Roboflow.

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F

from model import get_model, CLASS_NAMES

def auto_annotate(
    model_path='checkpoints/best_model.pth',
    data_dir='data',
    split='train',
    num_classes=7,
    score_threshold=0.3,
    output_dir='outputs/auto_annotations'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')

    # Load model
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Baca annotation file yang ada
    annotation_file = os.path.join(data_dir, split, '_annotations.coco.json')
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Cari gambar yang BELUM punya anotasi
    annotated_img_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    unannotated_images = [
        img for img in coco_data['images']
        if img['id'] not in annotated_img_ids
    ]

    print(f'[INFO] Total gambar di {split}: {len(coco_data["images"])}')
    print(f'[INFO] Sudah berlabel: {len(annotated_img_ids)}')
    print(f'[INFO] Belum berlabel: {len(unannotated_images)}')

    if len(unannotated_images) == 0:
        print('[INFO] Semua gambar sudah berlabel! Tidak ada yang perlu di-auto-annotate.')
        return

    # Jalankan prediksi pada gambar yang belum berlabel
    new_annotations = []
    ann_id_counter = 1
    images_with_predictions = 0

    print(f'\n[INFO] Memulai auto-annotation dengan score >= {score_threshold}...\n')

    with torch.no_grad():
        for img_info in tqdm(unannotated_images, desc='Auto-annotating'):
            img_path = os.path.join(data_dir, split, img_info['file_name'])
            if not os.path.exists(img_path):
                continue

            # Load dan preprocess gambar
            img = Image.open(img_path).convert('RGB')
            img_tensor = F.to_tensor(img).to(device)
            outputs = model([img_tensor])[0]

            boxes = outputs['boxes'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            masks = outputs['masks'].cpu().numpy()  # [N, 1, H, W]

            img_annotations = []

            for i in range(len(boxes)):
                score = float(scores[i])
                if score < score_threshold:
                    continue

                # Konversi bbox ke format COCO [x, y, w, h]
                x1, y1, x2, y2 = boxes[i]
                bbox_w = float(x2 - x1)
                bbox_h = float(y2 - y1)
                area = bbox_w * bbox_h

                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                # Konversi mask ke polygon (format COCO segmentation)
                binary_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                segmentation = mask_to_polygon(binary_mask)

                if not segmentation:
                    continue

                ann = {
                    'id': ann_id_counter,
                    'image_id': img_info['id'],
                    'category_id': int(labels[i]),
                    'bbox': [round(float(x1), 3), round(float(y1), 3),
                             round(bbox_w, 3), round(bbox_h, 3)],
                    'area': round(area, 3),
                    'segmentation': segmentation,
                    'score': round(score, 4),  # Skor confidence (untuk review)
                    'iscrowd': 0
                }
                img_annotations.append(ann)
                ann_id_counter += 1

            if img_annotations:
                images_with_predictions += 1
                new_annotations.extend(img_annotations)

    # Simpan hasil sebagai COCO JSON baru
    os.makedirs(output_dir, exist_ok=True)

    # 1. File khusus prediksi saja (untuk review)
    predicted_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': unannotated_images,
        'annotations': new_annotations
    }

    pred_path = os.path.join(output_dir, f'predicted_{split}.json')
    with open(pred_path, 'w') as f:
        json.dump(predicted_coco, f, indent=2)

    # 2. File merged (anotasi lama + prediksi baru) untuk langsung training
    # Renumber annotation IDs agar tidak bentrok
    max_existing_id = max((ann['id'] for ann in coco_data['annotations']), default=0)
    merged_annotations = list(coco_data['annotations'])
    for ann in new_annotations:
        merged_ann = dict(ann)
        merged_ann['id'] = max_existing_id + merged_ann['id']
        if 'score' in merged_ann:
            del merged_ann['score']  # Hapus score, tidak perlu di training
        merged_annotations.append(merged_ann)

    merged_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': coco_data['images'],
        'annotations': merged_annotations
    }

    merged_path = os.path.join(output_dir, f'merged_{split}.json')
    with open(merged_path, 'w') as f:
        json.dump(merged_coco, f, indent=2)

    # Ringkasan
    print(f'\n{"="*50}')
    print(f'  HASIL AUTO-ANNOTATION')
    print(f'{"="*50}')
    print(f'  Gambar diproses     : {len(unannotated_images)}')
    print(f'  Gambar ada prediksi : {images_with_predictions}')
    print(f'  Total anotasi baru  : {len(new_annotations)}')
    print(f'{"="*50}')
    print(f'\n  File output:')
    print(f'  1. {os.path.abspath(pred_path)}')
    print(f'     -> Hanya prediksi baru (untuk review di Roboflow)')
    print(f'  2. {os.path.abspath(merged_path)}')
    print(f'     -> Gabungan anotasi lama + prediksi baru')
    print(f'\n  Cara pakai merged file untuk training:')
    print(f'     Salin file merged ke data/{split}/_annotations.coco.json')
    print(f'     Lalu jalankan: py main.py --mode train')

    # Tampilkan distribusi kelas dari prediksi
    from collections import Counter
    pred_classes = Counter(ann['category_id'] for ann in new_annotations)
    print(f'\n  Distribusi prediksi per kelas:')
    for cid in sorted(pred_classes.keys()):
        name = CLASS_NAMES.get(cid, f'class_{cid}')
        print(f'    {name:<15s} : {pred_classes[cid]}')


def mask_to_polygon(binary_mask):
    """Konversi binary mask ke polygon COCO format menggunakan cv2"""
    import cv2
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # Minimal 3 titik (6 koordinat)
            segmentation.append(contour)
    return segmentation


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Auto-annotate unlabeled images')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path ke model')
    parser.add_argument('--split', default='train', help='Split dataset (train/valid/test)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Score threshold')
    args = parser.parse_args()

    auto_annotate(
        model_path=args.model,
        split=args.split,
        score_threshold=args.threshold
    )
