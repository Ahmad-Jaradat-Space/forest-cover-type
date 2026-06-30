"""Visuals for the land-cover segmentation notebook.

Shares the paper-and-ink aesthetic of plots.py and the class palette of
seg_data.py. The hero figure is the image / ground-truth / prediction
triptych; the per-class IoU bar is the one that tells the rare-class story.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import seg_data as sd
from plots import PAPER, INK, MUTED, PRIMARY, GOOD, WARN, ACCENT

_RGB = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                 for c in sd.CLASS_COLORS], dtype=np.uint8)


def colorize(mask):
    """Class-index mask (H,W) -> RGB image using the class palette."""
    return _RGB[mask]


def denorm(img_chw):
    """Undo ImageNet normalisation -> HWC uint8 for display."""
    x = img_chw.transpose(1, 2, 0) * sd.IMAGENET_STD + sd.IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def legend_handles():
    return [mpatches.Patch(color=c, label=n)
            for c, n in zip(sd.CLASS_COLORS, sd.CLASS_NAMES)]


def class_distribution_bar(fracs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    order = np.argsort(fracs)[::-1]
    bars = ax.bar([sd.CLASS_NAMES[i] for i in order], fracs[order] * 100,
                  color=[sd.CLASS_COLORS[i] for i in order],
                  edgecolor=INK, lw=0.5)
    ax.set_yscale("log")
    for b, i in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, fracs[i] * 100,
                f"{fracs[i] * 100:.1f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("Share of labelled pixels (log)")
    ax.set_title("Pixel-class imbalance: woodland & background dominate, buildings/roads are <2%",
                 color=INK)
    ax.set_ylim(top=fracs.max() * 100 * 2)
    return ax


def tile_gallery(dataset, n=6, seed=0, ncols=6):
    """Image tiles with their ground-truth masks beneath."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5))
    for col, i in enumerate(idx):
        img, mask = dataset[i]
        axes[0, col].imshow(denorm(img.numpy()))
        axes[1, col].imshow(colorize(mask.numpy()))
        for r in range(2):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    axes[0, 0].set_ylabel("aerial", fontsize=10)
    axes[1, 0].set_ylabel("labels", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("LandCover.ai: 0.25–0.5 m aerial tiles and their five-class masks",
                 color=INK, y=1.0)
    plt.tight_layout()
    return fig


def training_curves(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.4))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_loss"] for h in history], "-o", ms=3,
            color=PRIMARY, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy", color=PRIMARY)
    ax2 = ax.twinx()
    ax2.plot(ep, [h["val_miou"] for h in history], "-o", ms=3,
             color=ACCENT, label="Val mIoU")
    ax2.set_ylabel("Validation mIoU", color=ACCENT)
    ax2.grid(False)
    ax.set_title("Transfer-learned U-Net converges in a handful of epochs", color=INK)
    return ax


def per_class_iou_bar(iou, support_frac, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.2))
    order = np.argsort(iou)[::-1]
    bars = ax.bar([sd.CLASS_NAMES[i] for i in order], iou[order],
                  color=[sd.CLASS_COLORS[i] for i in order],
                  edgecolor=INK, lw=0.5)
    for b, i in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, iou[i] + 0.015,
                f"{iou[i]:.2f}\n({support_frac[i] * 100:.1f}%)",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("IoU (test)")
    ax.set_title("Per-class IoU vs pixel share — the rare classes are where it's hard",
                 color=INK)
    return ax


def prediction_triptych(model, dataset, idxs, device, max_rows=4):
    """Rows of aerial | ground truth | prediction for chosen tiles."""
    import torch
    model.eval()
    idxs = list(idxs)[:max_rows]
    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    titles = ["Aerial image", "Ground truth", "Prediction"]
    with torch.no_grad():
        for r, i in enumerate(idxs):
            img, mask = dataset[i]
            pred = model(img[None].to(device)).argmax(1)[0].cpu().numpy()
            panels = [denorm(img.numpy()), colorize(mask.numpy()), colorize(pred)]
            for c, panel in enumerate(panels):
                axes[r, c].imshow(panel)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(titles[c], color=INK, fontsize=11)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    return fig


def pixel_confusion(conf, ax=None):
    import seaborn as sns
    cm = conf / conf.sum(axis=1, keepdims=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=sd.CLASS_NAMES, yticklabels=sd.CLASS_NAMES,
                linewidths=0.5, linecolor=PAPER, cbar_kws={"label": "Row-normalised"},
                ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Pixel confusion (row-normalised)", color=INK)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax
