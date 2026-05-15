"""
plot_dataset_stats.py
Membuat grafik statistik distribusi label/anotasi sampah dari semua split dataset.
Hasil disimpan ke checkpoints/plots/
"""

import json
import os
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_DIR   = 'data'
OUTPUT_DIR = 'checkpoints/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = ['train', 'valid', 'test']

# Mapping kategori (sesuai dataset COCO)
CATEGORY_NAMES = {
    0: 'Trash',
    1: 'plastic_bag',
    2: 'plastic_wrapper',
}

# Warna per kategori (konsisten di semua grafik)
PALETTE = {
    'Trash'           : '#E8724A',
    'plastic_bag'     : '#4A90D9',
    'plastic_wrapper' : '#2ECC71',
}

# ── Load data semua split ─────────────────────────────────────────────────────
split_data   = {}   # split -> dict json
split_counts = {}   # split -> {cat_name: count}
split_img    = {}   # split -> {total, labeled, unlabeled}

for split in SPLITS:
    path = os.path.join(DATA_DIR, split, '_annotations.coco.json')
    if not os.path.exists(path):
        continue

    with open(path) as f:
        data = json.load(f)

    images      = data.get('images', [])
    annotations = data.get('annotations', [])
    categories  = {c['id']: c['name'] for c in data.get('categories', [])}

    # Hitung jumlah anotasi per kategori
    cat_counter = collections.Counter()
    for ann in annotations:
        cid = ann['category_id']
        cat_counter[categories.get(cid, f'class_{cid}')] += 1

    # Hitung gambar berlabel vs tidak
    img_id_to_anns = {img['id']: 0 for img in images}
    for ann in annotations:
        img_id_to_anns[ann['image_id']] += 1

    labeled   = sum(1 for v in img_id_to_anns.values() if v > 0)
    unlabeled = len(images) - labeled

    split_data[split]   = data
    split_counts[split] = dict(cat_counter)
    split_img[split]    = {
        'total'    : len(images),
        'labeled'  : labeled,
        'unlabeled': unlabeled,
        'anns'     : len(annotations),
    }

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

cat_names = list(CATEGORY_NAMES.values())
colors    = [PALETTE[c] for c in cat_names]

# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 1 — Distribusi objek per kategori per split (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

x      = np.arange(len(cat_names))
n_splits = len(SPLITS)
width  = 0.25
offsets = np.linspace(-(n_splits-1)/2, (n_splits-1)/2, n_splits) * width

split_colors_map = {'train': '#4A90D9', 'valid': '#E8724A', 'test': '#2ECC71'}

for i, split in enumerate(SPLITS):
    if split not in split_counts:
        continue
    counts = [split_counts[split].get(c, 0) for c in cat_names]
    bars = ax.bar(x + offsets[i], counts, width,
                  label=split.capitalize(),
                  color=split_colors_map[split],
                  edgecolor='white', linewidth=0.8, alpha=0.9)
    # Nilai di atas bar
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(cat_names, fontsize=11)
ax.set_ylabel('Jumlah Objek Teranotasi', fontsize=12)
ax.set_title('Distribusi Objek Sampah per Kategori dan Split Dataset',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-0.5, len(cat_names) - 0.5)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'stats_1_distribusi_per_kategori.png')
plt.savefig(out, dpi=150)
plt.close()
print(f'[OK] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 2 — Pie chart distribusi kategori (Train saja — data terbanyak)
# ═══════════════════════════════════════════════════════════════════════════════
train_counts = split_counts.get('train', {})
pie_labels   = [c for c in cat_names if train_counts.get(c, 0) > 0]
pie_vals     = [train_counts.get(c, 0) for c in pie_labels]
pie_colors   = [PALETTE[c] for c in pie_labels]

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct='%1.1f%%', startangle=140,
    pctdistance=0.75, labeldistance=1.08,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=11)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')

ax.set_title('Distribusi Anotasi per Kategori Sampah\n(Training Set)',
             fontsize=13, fontweight='bold')

# Legend dengan jumlah absolut
legend_labels = [f'{lbl} ({v})' for lbl, v in zip(pie_labels, pie_vals)]
ax.legend(wedges, legend_labels, loc='lower center',
          bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, framealpha=0.8)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'stats_2_pie_kategori_train.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 3 — Status Labeling per Split (Labeled vs Unlabeled)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

split_labels = [s.capitalize() for s in SPLITS if s in split_img]
labeled_vals   = [split_img[s]['labeled']   for s in SPLITS if s in split_img]
unlabeled_vals = [split_img[s]['unlabeled'] for s in SPLITS if s in split_img]

x2    = np.arange(len(split_labels))
width = 0.45

b1 = ax.bar(x2 - width/2, labeled_vals,   width, label='Berlabel',
            color='#2ECC71', edgecolor='white', linewidth=1)
b2 = ax.bar(x2 + width/2, unlabeled_vals, width, label='Tidak Berlabel',
            color='#E74C3C', edgecolor='white', linewidth=1, alpha=0.8)

for bar, val in zip(b1, labeled_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar, val in zip(b2, unlabeled_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(x2)
ax.set_xticklabels(split_labels, fontsize=12)
ax.set_ylabel('Jumlah Gambar', fontsize=12)
ax.set_title('Status Labeling Gambar per Split Dataset', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'stats_3_status_labeling.png')
plt.savefig(out, dpi=150)
plt.close()
print(f'[OK] {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFIK 4 — Dashboard Statistik Dataset (Gabungan semua panel)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# Panel A — Grouped bar distribusi kategori
ax0 = fig.add_subplot(gs[0, :])
for i, split in enumerate(SPLITS):
    if split not in split_counts:
        continue
    counts = [split_counts[split].get(c, 0) for c in cat_names]
    bars = ax0.bar(x + offsets[i], counts, width,
                   label=split.capitalize(),
                   color=split_colors_map[split],
                   edgecolor='white', linewidth=0.8, alpha=0.9)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(val), ha='center', va='bottom', fontsize=8, fontweight='bold')

ax0.set_xticks(x)
ax0.set_xticklabels(cat_names, fontsize=10)
ax0.set_ylabel('Jumlah Objek')
ax0.set_title('(A) Distribusi Objek per Kategori Sampah', fontweight='bold')
ax0.legend(fontsize=9)
ax0.set_xlim(-0.5, len(cat_names) - 0.5)
ax0.grid(True, alpha=0.3, ls='--')

# Panel B — Pie train
ax1 = fig.add_subplot(gs[1, 0])
wedges, texts, autotexts = ax1.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct='%1.1f%%', startangle=140,
    pctdistance=0.72, labeldistance=1.1,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=9)
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight('bold')
ax1.set_title('(B) Komposisi Kategori (Train)', fontweight='bold')

# Panel C — Status labeling
ax2 = fig.add_subplot(gs[1, 1])
b1 = ax2.bar(x2 - width/2, labeled_vals,   width, label='Berlabel',
             color='#2ECC71', edgecolor='white', linewidth=1)
b2 = ax2.bar(x2 + width/2, unlabeled_vals, width, label='Tidak Berlabel',
             color='#E74C3C', edgecolor='white', linewidth=1, alpha=0.8)
for bar, val in zip(b1, labeled_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(b2, unlabeled_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(split_labels, fontsize=11)
ax2.set_ylabel('Jumlah Gambar')
ax2.set_title('(C) Status Labeling per Split', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, ls='--')

fig.suptitle('Statistik Dataset Deteksi Sampah Pesisir — Mask R-CNN',
             fontsize=15, fontweight='bold', y=1.01)

out = os.path.join(OUTPUT_DIR, 'stats_4_dashboard.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] {out}')


# ── Cetak Ringkasan Teks ──────────────────────────────────────────────────────
print('\n======= RINGKASAN STATISTIK DATASET =======')
for split in SPLITS:
    if split not in split_img:
        continue
    info = split_img[split]
    pct  = info['labeled'] / info['total'] * 100 if info['total'] > 0 else 0
    print(f'\n[{split.upper()}]')
    print(f'  Total gambar    : {info["total"]}')
    print(f'  Berlabel        : {info["labeled"]} ({pct:.1f}%)')
    print(f'  Tidak berlabel  : {info["unlabeled"]}')
    print(f'  Total anotasi   : {info["anns"]}')
    if split in split_counts:
        for cat, cnt in sorted(split_counts[split].items(), key=lambda x: -x[1]):
            print(f'    {cat:<15}: {cnt}')

print('\nGrafik disimpan di: checkpoints/plots/')
print('===========================================')
