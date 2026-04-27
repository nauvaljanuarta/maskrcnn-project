"""
plot_training.py
Membuat grafik evaluasi training dari training_log.csv
Hasil disimpan ke folder checkpoints/plots/
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Konfigurasi ──────────────────────────────────────────────────────────────
LOG_PATH    = 'checkpoints/training_log.csv'
OUTPUT_DIR  = 'checkpoints/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
epochs, train_losses, val_losses, lrs = [], [], [], []

with open(LOG_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        train_losses.append(float(row['train_loss']))
        val_losses.append(float(row['val_loss']))
        lrs.append(float(row['lr']))

epochs       = np.array(epochs)
train_losses = np.array(train_losses)
val_losses   = np.array(val_losses)
lrs          = np.array(lrs)

best_epoch    = epochs[np.argmin(val_losses)]
best_val_loss = np.min(val_losses)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 11,
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'grid.linestyle'  : '--',
})

COLORS = {
    'train' : '#4A90D9',
    'val'   : '#E8724A',
    'best'  : '#2ECC71',
    'lr'    : '#9B59B6',
    'gap'   : '#F5A623',
}

# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 1 — Training vs Validation Loss
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, train_losses, color=COLORS['train'], lw=2.2, marker='o',
        markersize=5, label='Train Loss')
ax.plot(epochs, val_losses,   color=COLORS['val'],   lw=2.2, marker='s',
        markersize=5, label='Validation Loss')

# Highlight best
ax.axvline(best_epoch, color=COLORS['best'], ls='--', lw=1.5,
           label=f'Best Epoch ({best_epoch})  val_loss={best_val_loss:.4f}')
ax.scatter([best_epoch], [best_val_loss], color=COLORS['best'], zorder=5, s=80)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training vs Validation Loss (Mask R-CNN)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, '1_train_val_loss.png')
plt.savefig(out1, dpi=150)
plt.close()
print(f'[OK] Disimpan: {out1}')


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 2 — Overfitting Gap (Val Loss − Train Loss)
# ═══════════════════════════════════════════════════════════════════════════════
gap = val_losses - train_losses

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(epochs, gap, color=np.where(gap > 0, COLORS['gap'], COLORS['train']),
       alpha=0.8, edgecolor='white')
ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val Loss − Train Loss', fontsize=12)
ax.set_title('Generalization Gap per Epoch', fontsize=14, fontweight='bold')
ax.set_xlim(0.3, epochs[-1] + 0.7)

note = ("Bar positif (oranye) = model mulai overfit\n"
        "Bar negatif (biru) = model masih generalizing")
ax.text(0.99, 0.97, note, transform=ax.transAxes, fontsize=9,
        va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4',
        facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, '2_generalization_gap.png')
plt.savefig(out2, dpi=150)
plt.close()
print(f'[OK] Disimpan: {out2}')


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 3 — Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.step(epochs, lrs, color=COLORS['lr'], lw=2.2, where='mid', label='Learning Rate')
ax.fill_between(epochs, lrs, step='mid', alpha=0.15, color=COLORS['lr'])
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.legend(fontsize=10)

plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, '3_lr_schedule.png')
plt.savefig(out3, dpi=150)
plt.close()
print(f'[OK] Disimpan: {out3}')


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 4 — Dashboard gabungan (siap paste ke laporan)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# -- Panel A: train vs val loss
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(epochs, train_losses, color=COLORS['train'], lw=2.2, marker='o',
         markersize=5, label='Train Loss')
ax0.plot(epochs, val_losses,   color=COLORS['val'],   lw=2.2, marker='s',
         markersize=5, label='Validation Loss')
ax0.axvline(best_epoch, color=COLORS['best'], ls='--', lw=1.5,
            label=f'Best Epoch {best_epoch}  (val={best_val_loss:.4f})')
ax0.scatter([best_epoch], [best_val_loss], color=COLORS['best'], zorder=5, s=80)
ax0.set_title('(A) Training vs Validation Loss', fontweight='bold')
ax0.set_xlabel('Epoch'); ax0.set_ylabel('Loss')
ax0.legend(fontsize=9)
ax0.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
ax0.grid(True, alpha=0.3, ls='--')

# -- Panel B: Generalization Gap
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(epochs, gap, color=np.where(gap > 0, COLORS['gap'], COLORS['train']),
        alpha=0.85, edgecolor='white')
ax1.axhline(0, color='black', lw=1)
ax1.set_title('(B) Generalization Gap', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Val − Train Loss')
ax1.set_xlim(0.3, epochs[-1] + 0.7)
ax1.grid(True, alpha=0.3, ls='--')

# -- Panel C: LR
ax2 = fig.add_subplot(gs[1, 1])
ax2.step(epochs, lrs, color=COLORS['lr'], lw=2.2, where='mid')
ax2.fill_between(epochs, lrs, step='mid', alpha=0.15, color=COLORS['lr'])
ax2.set_title('(C) Learning Rate Schedule', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('LR')
ax2.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2.grid(True, alpha=0.3, ls='--')

fig.suptitle('Evaluasi Training Mask R-CNN — Deteksi Sampah Plastik',
             fontsize=15, fontweight='bold', y=1.01)

out4 = os.path.join(OUTPUT_DIR, '4_dashboard.png')
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Disimpan: {out4}')

# ── Ringkasan ─────────────────────────────────────────────────────────────────
print('\n========== RINGKASAN TRAINING ==========')
print(f'Total epoch berjalan : {len(epochs)}')
print(f'Best epoch           : {best_epoch}')
print(f'Best val loss        : {best_val_loss:.4f}')
print(f'Final train loss     : {train_losses[-1]:.4f}')
print(f'Final val loss       : {val_losses[-1]:.4f}')
print(f'Final LR             : {lrs[-1]:.6f}')
print(f'\nGrafik disimpan di   : checkpoints/plots/')
print('=========================================')
