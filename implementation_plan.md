# 📋 Panduan Implementasi Mask R-CNN — Skripsi Deteksi Sampah Drone

> **Framework**: PyTorch + torchvision | **GPU**: RTX 3050 | **OS**: Windows  
> Ikuti langkah-langkah di bawah ini secara berurutan.

---

## LANGKAH 1 — Install Environment

Buka **terminal / PowerShell / CMD** di folder proyekmu, lalu jalankan:

### 1.1 Install PyTorch (dengan CUDA untuk GPU)
Pergi ke https://pytorch.org/get-started/locally/ dan pilih:
- PyTorch: Stable
- OS: Windows
- Package: Pip
- Compute Platform: CUDA 11.8 atau 12.1 (sesuai driver GPU kamu)

Contoh command-nya (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 Buat file `requirements.txt`
Buat file ini di root folder proyek (`skripsiau/requirements.txt`):
```
pycocotools-windows   # khusus Windows (bukan pycocotools biasa!)
Pillow
numpy
matplotlib
tqdm
opencv-python
```

Lalu install:
```bash
pip install -r requirements.txt
```

> ⚠️ **Di Windows**, jangan install `pycocotools` biasa — pakai `pycocotools-windows`!

### 1.3 Cek apakah GPU terdeteksi
```python
import torch
print(torch.cuda.is_available())       # harus True
print(torch.cuda.get_device_name(0))   # harus "NVIDIA GeForce RTX 3050"
```

---

## LANGKAH 2 — Buat File `dataset.py`

File ini bertugas **load gambar + anotasi COCO** dan mengubahnya ke format yang bisa dibaca Mask R-CNN.

```python
# dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


class SampahDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        """
        root            : folder gambar (misal: 'data/train')
        annotation_file : path ke JSON COCO (misal: 'data/train/_annotations.coco.json')
        transforms      : augmentasi (opsional)
        """
        self.root = root
        self.transforms = transforms

        # Load JSON
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        # Buat mapping: image_id → info gambar
        self.img_id_to_info = {img['id']: img for img in coco['images']}

        # Buat mapping: image_id → list anotasi
        self.img_id_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        # List image_id yang punya anotasi
        self.img_ids = [
            img_id for img_id in self.img_id_to_info
            if img_id in self.img_id_to_anns
        ]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_id_to_info[img_id]
        anns = self.img_id_to_anns[img_id]

        # Load gambar
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        # Siapkan target
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            # Bounding box: COCO format [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue  # skip bbox tidak valid

            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])  # 1..6
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

            # Konversi polygon segmentasi → binary mask
            mask = self._poly_to_mask(ann['segmentation'], H, W)
            masks.append(mask)

        # Kalau tidak ada objek valid, skip (seharusnya tidak terjadi)
        if len(boxes) == 0:
            return None

        target = {
            'boxes'   : torch.as_tensor(boxes,    dtype=torch.float32),
            'labels'  : torch.as_tensor(labels,   dtype=torch.int64),
            'masks'   : torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'image_id': torch.tensor([img_id]),
            'area'    : torch.as_tensor(areas,    dtype=torch.float32),
            'iscrowd' : torch.as_tensor(iscrowd,  dtype=torch.int64),
        }

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _poly_to_mask(self, segmentation, H, W):
        """Konversi polygon COCO → numpy binary mask (H, W)"""
        import numpy as np
        from PIL import ImageDraw

        mask_img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img)
        for poly in segmentation:
            # poly adalah list [x1,y1,x2,y2,...] → ubah ke [(x1,y1),(x2,y2),...]
            points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            draw.polygon(points, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)


def collate_fn(batch):
    """Custom collate untuk DataLoader — filter None dan gabungkan batch."""
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


class SimpleTransform:
    """Transform sederhana: PIL Image → Tensor, opsional horizontal flip."""
    def __init__(self, train=False):
        self.train = train

    def __call__(self, image, target):
        import random
        # Random horizontal flip (hanya saat training)
        if self.train and random.random() > 0.5:
            W = image.width
            image = F.hflip(image)
            # Flip boxes
            boxes = target['boxes']
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            target['boxes'] = boxes
            # Flip masks
            target['masks'] = target['masks'].flip(-1)

        # Ubah gambar ke tensor [C, H, W]
        image = F.to_tensor(image)
        return image, target
```

---

## LANGKAH 3 — Buat File `model.py`

File ini mendefinisikan **Mask R-CNN** yang sudah dimodifikasi untuk 6 kelas sampahmu.

```python
# model.py
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes=7):
    """
    Buat Mask R-CNN dengan pretrained weights COCO,
    lalu ganti head-nya agar output = num_classes.

    num_classes = 7 → (background + 6 kelas sampah)
    """
    # Load Mask R-CNN pretrained COCO
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # ── Ganti Box Predictor ──────────────────────────────────────
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # ── Ganti Mask Predictor ─────────────────────────────────────
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,
        num_classes=num_classes
    )

    return model


# ══ Mapping kelas (untuk visualisasi) ══════════════════════════
CLASS_NAMES = {
    0: 'background',
    1: 'Cloth',
    2: 'Foam',
    3: 'Hard Plastic',
    4: 'Other',
    5: 'Paper',
    6: 'Soft Plastic',
}

CLASS_COLORS = {
    1: (0,   114, 189),   # Cloth        → Biru
    2: (50,  205,  50),   # Foam         → Hijau
    3: (217,  83,  25),   # Hard Plastic → Merah-Oranye
    4: (237, 177,  32),   # Other        → Kuning
    5: (126,  47, 142),   # Paper        → Ungu
    6: (255, 140,   0),   # Soft Plastic → Oranye
}
```

---

## LANGKAH 4 — Buat File `train.py`

File utama untuk **melatih** model Mask R-CNN.

```python
# train.py
import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SampahDataset, SimpleTransform, collate_fn
from model import get_model

# ══ KONFIGURASI — ubah sesuai kebutuhan ══════════════════════════
CONFIG = {
    'data_dir'     : 'data',
    'num_classes'  : 7,          # 6 kelas + 1 background
    'num_epochs'   : 30,
    'batch_size'   : 2,          # RTX 3050 (4GB VRAM)
    'lr'           : 0.005,
    'momentum'     : 0.9,
    'weight_decay' : 0.0005,
    'step_size'    : 10,         # turunkan LR setiap 10 epoch
    'gamma'        : 0.1,
    'num_workers'  : 2,
    'save_dir'     : 'checkpoints',
    'print_freq'   : 10,         # print loss setiap 10 iterasi
}


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    n_batches = len(data_loader)

    loop = tqdm(data_loader, desc=f'Epoch [{epoch}]')
    for i, (images, targets) in enumerate(loop):
        # Pindah ke GPU
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass → hitung semua losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())

    avg_loss = total_loss / n_batches
    return avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Menggunakan device: {device}')

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # ── Dataset & DataLoader ────────────────────────────────────
    train_dataset = SampahDataset(
        root=os.path.join(CONFIG['data_dir'], 'train'),
        annotation_file=os.path.join(CONFIG['data_dir'], 'train', '_annotations.coco.json'),
        transforms=SimpleTransform(train=True)
    )
    valid_dataset = SampahDataset(
        root=os.path.join(CONFIG['data_dir'], 'valid'),
        annotation_file=os.path.join(CONFIG['data_dir'], 'valid', '_annotations.coco.json'),
        transforms=SimpleTransform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn
    )

    print(f'[INFO] Train: {len(train_dataset)} gambar')
    print(f'[INFO] Valid: {len(valid_dataset)} gambar')

    # ── Model ────────────────────────────────────────────────────
    model = get_model(num_classes=CONFIG['num_classes'])
    model.to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG['lr'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['step_size'],
        gamma=CONFIG['gamma']
    )

    # ── Log file ─────────────────────────────────────────────────
    log_path = os.path.join(CONFIG['save_dir'], 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'lr'])

    best_loss = float('inf')

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:02d} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}')

        # Simpan log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(train_loss, 4), current_lr])

        # Simpan model terbaik
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print(f'  → Best model saved! (loss={best_loss:.4f})')

        # Simpan model terakhir
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'last_model.pth'))

    print(f'\n[DONE] Training selesai! Best loss: {best_loss:.4f}')
    print(f'[DONE] Model tersimpan di: {CONFIG["save_dir"]}/')


if __name__ == '__main__':
    main()
```

**Cara jalankan:**
```bash
python train.py
```

---

## LANGKAH 5 — Buat File `evaluate.py`

Script untuk **mengevaluasi** model menggunakan COCO metrics (mAP).

```python
# evaluate.py
import os
import json
import torch
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
                    rle = maskutil.encode(binary_mask)
                    rle['counts'] = rle['counts'].decode('utf-8')

                    results_segm.append({
                        'image_id'   : image_id,
                        'category_id': int(labels[i]),
                        'segmentation': rle,
                        'score'      : score,
                    })

    # Evaluasi dengan pycocotools
    coco_gt = COCO(annotation_file)

    print('\n══════════════ BBOX mAP ══════════════')
    if results_bbox:
        coco_dt = coco_gt.loadRes(results_bbox)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    print('\n══════════════ SEGMENTATION mAP ══════════════')
    if results_segm:
        coco_dt = coco_gt.loadRes(results_segm)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    evaluate(
        model_path='checkpoints/best_model.pth',
        data_dir='data',
        split='test'
    )
```

**Cara jalankan:**
```bash
python evaluate.py
```

---

## LANGKAH 6 — Buat File `predict.py`

Script untuk **prediksi gambar baru** dan visualisasi hasilnya.

```python
# predict.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as F
import random

from model import get_model, CLASS_NAMES, CLASS_COLORS


def predict_image(image_path, model_path, num_classes=7, score_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load gambar
    image = Image.open(image_path).convert('RGB')
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Prediksi
    with torch.no_grad():
        output = model(img_tensor)[0]

    boxes  = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks  = output['masks'].cpu().numpy()  # [N, 1, H, W]

    # Filter berdasarkan score threshold
    keep = scores >= score_threshold
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks  = masks[keep]

    print(f'Ditemukan {len(boxes)} objek sampah:')
    for i in range(len(boxes)):
        cls = CLASS_NAMES.get(labels[i], 'unknown')
        print(f'  [{i+1}] {cls} — confidence: {scores[i]:.2%}')

    # Visualisasi
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f'Hasil Deteksi: {len(boxes)} objek ditemukan')

    img_np = np.array(image).copy()

    for i in range(len(boxes)):
        label_id = labels[i]
        cls_name = CLASS_NAMES.get(label_id, 'unknown')
        color = CLASS_COLORS.get(label_id, (255, 255, 255))
        color_norm = tuple(c/255 for c in color)

        # Gambar bounding box
        x1, y1, x2, y2 = boxes[i]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color_norm, facecolor='none'
        )
        ax.add_patch(rect)

        # Label + score
        ax.text(
            x1, y1 - 5,
            f'{cls_name} {scores[i]:.0%}',
            color='white',
            fontsize=9,
            bbox=dict(facecolor=color_norm, alpha=0.8, pad=2)
        )

        # Gambar mask semi-transparan
        mask = masks[i, 0]  # [H, W]
        mask_bin = mask > 0.5
        overlay = np.zeros_like(img_np, dtype=np.uint8)
        overlay[mask_bin] = color
        img_np = np.where(
            mask_bin[:, :, None],
            (img_np * 0.6 + overlay * 0.4).astype(np.uint8),
            img_np
        )

    ax.imshow(img_np)
    ax.axis('off')

    out_path = image_path.replace('.jpg', '_predicted.jpg').replace('.png', '_predicted.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f'\nHasil disimpan: {out_path}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python predict.py path/to/image.jpg')
    else:
        predict_image(
            image_path=sys.argv[1],
            model_path='checkpoints/best_model.pth',
            score_threshold=0.5
        )
```

**Cara jalankan:**
```bash
python predict.py data/test/NAMA_GAMBAR.jpg
```

---

## LANGKAH 7 — Update `main.py`

Entry point untuk semua mode.

```python
# main.py
import argparse


def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN — Deteksi Sampah Drone')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'predict'],
                        help='Mode: train | eval | predict')
    parser.add_argument('--image', type=str, default=None,
                        help='Path gambar untuk mode predict')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path model .pth')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold untuk predict (default: 0.5)')
    args = parser.parse_args()

    if args.mode == 'train':
        print('[MODE] Training...')
        from train import main as train_main
        train_main()

    elif args.mode == 'eval':
        print('[MODE] Evaluasi...')
        from evaluate import evaluate
        evaluate(model_path=args.model, data_dir='data', split='test')

    elif args.mode == 'predict':
        if not args.image:
            print('[ERROR] Tambahkan --image path/gambar.jpg')
        else:
            print(f'[MODE] Prediksi: {args.image}')
            from predict import predict_image
            predict_image(
                image_path=args.image,
                model_path=args.model,
                score_threshold=args.threshold
            )


if __name__ == '__main__':
    main()
```

---

## URUTAN PENGERJAAN (Ringkasan)

```
1. ✅ Install PyTorch + dependencies (LANGKAH 1)
2. ✅ Buat dataset.py  (LANGKAH 2)
3. ✅ Buat model.py    (LANGKAH 3)
4. ✅ Buat train.py    (LANGKAH 4)
5. ✅ Buat evaluate.py (LANGKAH 5)
6. ✅ Buat predict.py  (LANGKAH 6)
7. ✅ Update main.py   (LANGKAH 7)

Jalankan:
8. python train.py           → training (~2-3 jam)
9. python evaluate.py        → evaluasi mAP
10. python predict.py data/test/NAMA.jpg → prediksi
```

---

## Tips & Troubleshooting

### ❌ Error: `CUDA out of memory`
→ Kurangi `batch_size` di `train.py` dari `2` menjadi `1`

### ❌ Error: `pycocotools` tidak bisa install
→ Pastikan pakai `pip install pycocotools-windows` (bukan `pycocotools`)

### ❌ Error: `No module named 'cv2'`
→ Jalankan `pip install opencv-python`

### ❌ Training loss tidak turun setelah banyak epoch
→ Coba kurangi learning rate: ubah `lr` dari `0.005` ke `0.001`

### ✅ Cek apakah GPU dipakai saat training
→ Buka Task Manager Windows → tab Performance → GPU → lihat GPU Engine (`3D`) aktif

---

## Struktur Folder Akhir

```
skripsiau/
├── data/
│   ├── train/        ← gambar + _annotations.coco.json
│   ├── valid/        ← gambar + _annotations.coco.json
│   └── test/         ← gambar + _annotations.coco.json
├── checkpoints/      ← dibuat otomatis oleh train.py
│   ├── best_model.pth
│   ├── last_model.pth
│   └── training_log.csv
├── requirements.txt
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
└── main.py
```
