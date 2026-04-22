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
