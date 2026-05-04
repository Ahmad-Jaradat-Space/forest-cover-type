"""Plot helpers used by the notebook."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

PALETTE = sns.color_palette("Set2", 7)
ACCENT = "#2c5f8d"
WARN = "#c44e52"
CONTEXT = "#7f7f7f"


def caption(fig, text):
    """One-line takeaway caption under a figure."""
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def annotate_point(ax, x, y, text, dx=20, dy=20):
    """Arrow + label pointing at (x, y) on `ax`."""
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#bbb", lw=0.6, alpha=0.9),
    )


def apply_style():
    sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "font.family": "sans-serif",
    })


def class_balance_bar(y, class_names, ax=None):
    counts = np.bincount(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(class_names, counts, color=PALETTE)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("training examples")
    ax.set_title("class distribution")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ax


def confusion(y_true, y_pred, class_names, title, ax=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    cm_show = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_show, annot=cm, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.4, linecolor="white", ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax


def loss_curve(history, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history],
            "-o", markersize=3, label="train", color=ACCENT)
    if "val_loss" in history[0]:
        ax.plot(epochs, [h["val_loss"] for h in history],
                "-o", markersize=3, label="val", color=WARN)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
    ax.legend()
    return ax


def learning_curve(sizes, train_scores, val_scores, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, train_scores, "-o", label="train", color=ACCENT)
    ax.plot(sizes, val_scores, "-o", label="val", color=WARN)
    ax.fill_between(sizes, train_scores, val_scores,
                    alpha=0.1, color="grey", label="generalisation gap")
    ax.set_xscale("log")
    ax.set_xlabel("training set size (log)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    return ax


def feature_importance(names, importances, top=15, ax=None):
    order = np.argsort(importances)[::-1][:top]
    names_sorted = np.array(names)[order][::-1]
    vals = importances[order][::-1]

    # colour by category
    def cat_color(name):
        if name.startswith("wilderness"):
            return "#8da0cb"
        if name.startswith("soil"):
            return "#a6d854"
        return ACCENT

    colors = [cat_color(n) for n in names_sorted]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(top), vals, color=colors)
    ax.set_yticks(range(top))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel("importance")
    ax.set_title("feature importance (random forest)")
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=ACCENT, label="continuous"),
        Patch(color="#8da0cb", label="wilderness area"),
        Patch(color="#a6d854", label="soil type"),
    ], loc="lower right")
    return ax


def pca_scatter(X2, y, class_names, title, ax=None, s=4):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for c in np.unique(y):
        m = y == c
        ax.scatter(X2[m, 0], X2[m, 1], s=s, alpha=0.4,
                   label=class_names[c], color=PALETTE[c])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(markerscale=4, fontsize=8, loc="best", framealpha=0.85)
    return ax


def per_class_recall(y_true, y_pred, class_names, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    rec = cm.diagonal() / cm.sum(axis=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(class_names, rec, color=PALETTE)
    for b, r in zip(bars, rec):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.01,
                f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall")
    ax.set_title("per-class recall (test set)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ax


def model_comparison(df, ax=None):
    """Horizontal grouped bar chart of accuracy and macro F1 per model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    df = df.sort_values("accuracy")
    y = np.arange(len(df))
    h = 0.4
    ax.barh(y - h / 2, df["accuracy"], height=h, label="accuracy", color=ACCENT)
    ax.barh(y + h / 2, df["macro F1"], height=h, label="macro F1", color=WARN)
    for i, (a, f) in enumerate(zip(df["accuracy"], df["macro F1"])):
        ax.text(a + 0.005, i - h / 2, f"{a:.2f}", va="center", fontsize=8)
        ax.text(f + 0.005, i + h / 2, f"{f:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("score")
    ax.set_xlim(0, 1.05)
    ax.set_title("test-set scores by model")
    ax.legend(loc="lower right")
    return ax
