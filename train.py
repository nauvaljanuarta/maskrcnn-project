# train.py
import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SampahDataset, SimpleTransform, collate_fn
from model import get_model

CONFIG = {
    'data_dir'     : 'data',
    'num_classes'  : 7,          
    'num_epochs'   : 45,
    'batch_size'   : 2,          
    'lr'           : 0.001,
    'momentum'     : 0.9,
    'weight_decay' : 0.0005,
    'step_size'    : 30,         
    'gamma'        : 0.1,
    'num_workers'  : 0,
    'save_dir'     : 'checkpoints',
    'print_freq'   : 10,         
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
    """Evaluasi validation loss tanpa update gradien"""
    model.train() # Harus mode train agar MaskRCNN mengembalikan nilai loss
    total_loss = 0
    n_batches = len(data_loader)
    
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['step_size'],
        gamma=CONFIG['gamma']
    )

    # ── Log file ─────────────────────────────────────────────────
    log_path = os.path.join(CONFIG['save_dir'], 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    # ── Training Loop ─────────────────────────────────────────────
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # 1. Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 2. Validasi
        val_loss = evaluate_loss(model, valid_loader, device)
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
            print(f'  → Best model saved! (val_loss={best_val_loss:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  → Early stopping counter: {epochs_no_improve}/{patience}')

        # Simpan model terakhir
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'last_model.pth'))

        # Cek Early Stopping
        if epochs_no_improve >= patience:
            print(f'\n[STOP] Early stopping memicu penghentian pada epoch {epoch}!')
            print(f'Validation loss tidak membaik selama {patience} epoch berturut-turut.')
            break

    print(f'\n[DONE] Training selesai! Best validation loss: {best_val_loss:.4f}')
    print(f'[DONE] Model tersimpan di: {CONFIG["save_dir"]}/')

if __name__ == '__main__':
    main()
