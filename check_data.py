import json
import os

base_data_dir = 'c:\\Folder Janu\\Assignment\\CODE\\skripsiau\\data'
splits = ['train', 'valid', 'test']

for split in splits:
    print(f"=== Memeriksa dataset: {split.upper()} ===")
    ann_file = os.path.join(base_data_dir, split, '_annotations.coco.json')
    
    if not os.path.exists(ann_file):
        print(f"File anotasi tidak ditemukan: {ann_file}\n")
        continue

    with open(ann_file, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])

    # Buat pemetaan (mapping) ID gambar ke daftar anotasinya
    img_id_to_anns = {img['id']: [] for img in images}
    for ann in annotations:
        img_id_to_anns[ann['image_id']].append(ann)

    unlabeled_images = []
    null_bbox_anns = []
    null_segmentation_anns = []

    # Cek gambar mana saja yang tidak punya anotasi (unlabeled)
    for img in images:
        if len(img_id_to_anns[img['id']]) == 0:
            unlabeled_images.append(img['file_name'])

    # Cek apakah ada anotasi yang format datanya corrupt/null
    for ann in annotations:
        # Cek kelengkapan bbox (harus [x, y, w, h])
        if 'bbox' not in ann or ann['bbox'] is None or len(ann['bbox']) != 4:
            null_bbox_anns.append(ann['id'])
        
        # Cek kelengkapan koordinat segmentasi polygon
        if 'segmentation' not in ann or ann['segmentation'] is None or len(ann['segmentation']) == 0:
            null_segmentation_anns.append(ann['id'])

    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Images without annotations (unlabeled): {len(unlabeled_images)}")
    print(f"Annotations with invalid bbox: {len(null_bbox_anns)}")
    print(f"Annotations with invalid segmentation: {len(null_segmentation_anns)}")

    if len(unlabeled_images) > 0:
        print(f"Example unlabeled images: {unlabeled_images[:3]}")
        
    if len(null_bbox_anns) > 0:
        print(f"Example invalid bbox annotation IDs: {null_bbox_anns[:3]}")

    if len(null_segmentation_anns) > 0:
        print(f"Example invalid segmentation annotation IDs: {null_segmentation_anns[:3]}")
        
    print("\n")
