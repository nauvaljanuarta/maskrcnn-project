# predict.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as F
import random
import os

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

    # Buat output foldernya
    os.makedirs('outputs', exist_ok=True)
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    out_path = os.path.join('outputs', f"{base_name}_predicted{ext}")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    # Tampilkan hasil gambar dan tunggu user menutupnya
    plt.show()
    
    print(f'\nHasil disimpan di: {out_path}')


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
