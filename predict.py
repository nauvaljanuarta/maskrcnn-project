# predict.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as F
import os
import argparse
from glob import glob

from model import get_model, CLASS_NAMES, CLASS_COLORS

def load_prediction_model(model_path, num_classes=4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(num_classes=num_classes)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model successfully loaded from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using uninitialized/default weights.")
    model.to(device)
    model.eval()
    return model

def predict_single_image(image_path, model, device, score_threshold=0.5, output_dir='outputs', show_plot=False):
    # Load gambar
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Prediksi
    with torch.no_grad():
        output = model(img_tensor)[0]

    boxes  = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    # Filter berdasarkan score threshold
    keep = scores >= score_threshold
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # Visualisasi
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f'Hasil Deteksi: {len(boxes)} objek ditemukan')

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

    ax.axis('off')

    # Buat output foldernya
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    out_path = os.path.join(output_dir, f"{base_name}_predicted{ext}")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return labels, out_path
 
def main():
    parser = argparse.ArgumentParser(description="Bulk Predict Bounding Boxes for Images")
    parser.add_argument('--input', type=str, default='data_predict', 
                        help='Path to file or folder containing images to predict (default: data_predict)')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Confidence score threshold (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions',
                        help='Directory to save prediction results (default: outputs/predictions)')
    parser.add_argument('--show', action='store_true',
                        help='Show prediction plot window during prediction (only recommended for single images)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model once
    model = load_prediction_model(args.model_path, num_classes=4, device=device)
    
    # Cari image path(s)
    image_paths = []
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # Cari secara rekursif ekstensi gambar populer
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        for ext in extensions:
            image_paths.extend(glob(os.path.join(args.input, '**', ext), recursive=True))
        # Hapus duplikat
        image_paths = sorted(list(set(image_paths)))
    else:
        print(f"Error: Input path '{args.input}' not found.")
        return

    if not image_paths:
        print(f"No images found in path '{args.input}'.")
        return

    print(f"Found {len(image_paths)} image(s) to process.")
    
    # Inisialisasi counter evaluasi tiap kelas
    class_counts = {name: 0 for cid, name in CLASS_NAMES.items() if cid != 0}
    total_detected = 0
    success_count = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {img_path}")
        result = predict_single_image(
            image_path=img_path,
            model=model,
            device=device,
            score_threshold=args.score_threshold,
            output_dir=args.output_dir,
            show_plot=args.show
        )
        if result is not None:
            detected_labels, out_path = result
            num_boxes = len(detected_labels)
            print(f"    -> Detected {num_boxes} objects. Saved to {out_path}")
            
            # Hitung per class
            for lid in detected_labels:
                cls_name = CLASS_NAMES.get(lid, 'unknown')
                if cls_name != 'background':
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_detected += 1
            success_count += 1
            
    print(f"\nCompleted! Successfully predicted {success_count}/{len(image_paths)} images.")
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Confidence Threshold: {args.score_threshold}")
    print(f"Total Objects Detected: {total_detected}")
    for cls_name, count in class_counts.items():
        print(f" - {cls_name}: {count} objects")
    print("====================================================")
    
    # Simpan hasil evaluasi summary ke file txt
    summary_path = os.path.join(args.output_dir, 'prediction_summary.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write("================ EVALUATION SUMMARY ================\n")
            f.write(f"Confidence Threshold: {args.score_threshold}\n")
            f.write(f"Total Images Processed: {len(image_paths)}\n")
            f.write(f"Successfully Predicted: {success_count}\n")
            f.write(f"Total Objects Detected: {total_detected}\n\n")
            f.write("Detected counts per class:\n")
            for cls_name, count in class_counts.items():
                f.write(f" - {cls_name}: {count} objects\n")
            f.write("====================================================\n")
        print(f"Evaluation summary saved to: {os.path.abspath(summary_path)}")
    except Exception as e:
        print(f"Error saving summary file: {e}")

    # Buat dan simpan grafik hasil evaluasi
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Ambil warna yang sesuai dengan CLASS_COLORS di model.py
        bar_colors = []
        for cls in classes:
            cid = None
            for k, v in CLASS_NAMES.items():
                if v == cls:
                    cid = k
                    break
            color = CLASS_COLORS.get(cid, (128, 128, 128))
            bar_colors.append(tuple(c/255 for c in color))
            
        bars = ax.bar(classes, counts, color=bar_colors, edgecolor='black', linewidth=1.2)
        
        # Tambahkan nilai di atas setiap batang grafik
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset 3 points vertikal
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
                        
        ax.set_title(f'Summary Evaluasi Deteksi Sampah\n(Threshold: {args.score_threshold})', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Kategori Sampah', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Jumlah Terdeteksi (Objek)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        chart_path = os.path.join(args.output_dir, 'prediction_summary_chart.png')
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Evaluation summary chart saved to: {os.path.abspath(chart_path)}")
    except Exception as e:
        print(f"Error generating or saving chart: {e}")

if __name__ == '__main__':
    main()

