# train.py
import os
import csv
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SampahDataset, SimpleTransform, collate_fn
from model import get_model

CONFIG = {
    'data_dir'     : 'data',
    'num_classes'  : 4,           # 3 kategori + 1 background
    'num_epochs'   : 100,         # lebih banyak epoch untuk dataset kecil
    'batch_size'   : 4,           # batch lebih besar → gradien lebih stabil
    'lr'           : 0.005,       # LR moderat untuk fine-tuning pretrained
    'momentum'     : 0.9,
    'weight_decay' : 0.0005,
    'warmup_epochs': 5,           # warmup 5 epoch pertama
    'num_workers'  : 0,
    'save_dir'     : 'checkpoints',
    'print_freq'   : 10,
    'patience'     : 15,          # early stopping: berhenti jika val_loss tidak membaik selama 15 epoch
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

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())

    avg_loss = total_loss / n_batches
    return avg_loss

def evaluate_loss(model, data_loader, device):
    model.train() 
    total_loss = 0
    n_batches = len(data_loader)
    
    torch.cuda.empty_cache()  
    with torch.no_grad():
        for images, targets in data_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / n_batches if n_batches > 0 else 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Menggunakan device: {device}')

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

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

    model = get_model(num_classes=CONFIG['num_classes'])
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG['lr'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    # Cosine Annealing — LR turun halus dari lr_max ke ~0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'] - CONFIG['warmup_epochs'],
        eta_min=1e-6
    )

    # ── Log file ─────────────────────────────────────────────────
    log_path = os.path.join(CONFIG['save_dir'], 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    best_val_loss = float('inf')
    epochs_no_improve = 0  # counter early stopping

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # ── Warmup: naikkan LR secara linear di awal ──
        if epoch <= CONFIG['warmup_epochs']:
            warmup_lr = CONFIG['lr'] * (epoch / CONFIG['warmup_epochs'])
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # 1. Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 2. Validasi
        val_loss = evaluate_loss(model, valid_loader, device)

        # Step scheduler hanya setelah warmup selesai
        if epoch > CONFIG['warmup_epochs']:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}')

        # Simpan log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(train_loss, 4), round(val_loss, 4), current_lr])

        # Simpan model terbaik BERSANDARKAN VALIDATION LOSS
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print(f'  -> Best model saved! (val_loss={best_val_loss:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  -> No improvement ({epochs_no_improve}/{CONFIG["patience"]})')

        # Simpan model terakhir
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'last_model.pth'))

        # ── Early Stopping ──
        if epochs_no_improve >= CONFIG['patience']:
            print(f'\n[EARLY STOP] Val loss tidak membaik selama {CONFIG["patience"]} epoch. Berhenti di epoch {epoch}.')
            break

    print(f'\n[DONE] Training selesai! Best validation loss: {best_val_loss:.4f}')
    print(f'[DONE] Model tersimpan di: {CONFIG["save_dir"]}/')

if __name__ == '__main__':
    main()
