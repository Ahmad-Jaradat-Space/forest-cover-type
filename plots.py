"""Production-grade plot helpers for the forest-cover-type notebook."""

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support)

# Design palette (shared across the capstone series)
PRIMARY = "#0E4F5F"
ACCENT  = "#D88C4A"
GOOD    = "#5C9D7E"
WARN    = "#B0413E"
MUTED   = "#7A8C99"
INK     = "#1A1A1A"
PAPER   = "#FAFAF7"

# Per-class palette (7 cover types). Ordered by frequency in the dataset.
PALETTE = [
    "#0E4F5F",  # 1 Spruce/Fir
    "#D88C4A",  # 2 Lodgepole Pine
    "#5C9D7E",  # 3 Ponderosa
    "#5E548E",  # 4 Cottonwood/Willow
    "#B0413E",  # 5 Aspen
    "#1F6F8B",  # 6 Douglas Fir
    "#A66E38",  # 7 Krummholz
]
CONTEXT = MUTED


def apply_style():
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.titlepad": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#BFC4CA",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "font.size": 10.5,
    })
    try:
        mpl.rcParams["text.parse_math"] = False
    except KeyError:
        pass


# ------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------
def caption(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def annotate_point(ax, x, y, text, dx=20, dy=20, color=INK):
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec=color, lw=0.6, alpha=0.95),
    )


def kpi_card(ax, value, label, sub=None, color=PRIMARY):
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=22, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center",
            fontsize=10, color=INK, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.12, sub, ha="center", va="center",
                fontsize=8.5, color=MUTED, transform=ax.transAxes, style="italic")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, ec="#D8DCE2", lw=1.0))


def kpi_banner(values):
    fig, axes = plt.subplots(1, len(values), figsize=(2.8 * len(values), 1.8))
    if len(values) == 1:
        axes = [axes]
    palette = [PRIMARY, ACCENT, WARN, GOOD, "#5E548E"]
    for ax, v, color in zip(axes, values, palette):
        kpi_card(ax, *v, color=color)
    plt.tight_layout()
    return fig


def business_summary(rows, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1))
    ax.axis("off")
    n = len(rows)
    for i, (lhs, rhs) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax.text(0.02, y, lhs, transform=ax.transAxes,
                fontsize=10.5, color=INK, fontweight="bold", va="center")
        ax.text(0.30, y, rhs, transform=ax.transAxes,
                fontsize=10, color=INK, va="center")
        ax.plot([0.01, 0.99], [1 - i / n, 1 - i / n],
                color="#E0E4EA", lw=0.6, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    return ax


# ------------------------------------------------------------------
# Class balance + features
# ------------------------------------------------------------------
def class_balance_bar(y, class_names, ax=None):
    counts = np.bincount(y)[1:] if (y.min() >= 1) else np.bincount(y)
    if len(counts) == 0:
        counts = np.bincount(y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(class_names, counts, color=PALETTE[:len(class_names)],
                  edgecolor=INK, lw=0.5, alpha=0.92)
    ax.set_yscale("log")
    log_ticks = [10 ** k for k in range(int(np.log10(counts.max())) + 2)]
    ax.set_yticks(log_ticks)
    ax.set_yticklabels(
        [f"{t:,}" if t < 1000 else f"{t//1000}K" if t < 1_000_000 else f"{t//1_000_000}M"
         for t in log_ticks])
    for b, c in zip(bars, counts):
        rate = c / counts.sum() * 100
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({rate:.1f}%)",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Training examples (log)")
    ax.set_title("Class imbalance: top two species hold ~85% of patches",
                 color=INK)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylim(top=counts.max() * 5)
    return ax


def feature_distributions_grid(X, y, feat_names, cols, class_names, ax_row=None):
    """Per-feature small-multiples; histograms by class with shared y-axis."""
    n = len(cols)
    if ax_row is None:
        fig, ax_row = plt.subplots(1, n, figsize=(4.5 * n, 3.4), sharey=True)
    if n == 1:
        ax_row = [ax_row]
    for ax, col_idx, col_name in zip(ax_row, cols, [feat_names[c] for c in cols]):
        for c, color, name in zip(np.unique(y), PALETTE, class_names):
            ax.hist(X[y == c, col_idx], bins=50, alpha=0.45,
                    color=color, label=name, density=True,
                    edgecolor=PAPER, lw=0.3)
        ax.set_xlabel(col_name)
        ax.set_title(col_name, color=INK, fontsize=11)
    ax_row[0].set_ylabel("Density")
    ax_row[-1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    return ax_row


# ------------------------------------------------------------------
# Confusion / per-class metrics
# ------------------------------------------------------------------
def confusion(y_true, y_pred, class_names, title, ax=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    cm_show = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm_show, annot=False, cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor=PAPER, ax=ax,
        vmin=0, vmax=1 if normalize else None,
        cbar_kws={"label": "Row-normalised proportion" if normalize else "Count"},
    )
    cmax = cm_show.max() if normalize else cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            cell = cm_show[i, j]
            txt_color = PAPER if cell > 0.55 * cmax else INK
            label = f"{cell:.2f}" if normalize else f"{int(v)}"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=txt_color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, color=INK)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax


def confusion_with_f1_strip(y_true, y_pred, class_names, title, ax=None):
    """Confusion matrix with a one-column F1 strip on the right.

    Highlights minority-class performance directly next to the confusion block —
    so a 0.18 F1 next to a 0.84 row of off-diagonal mass is impossible to miss."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    n = len(class_names)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 labels=range(1, n + 1) if y_true.min() >= 1 else range(n),
                                                 zero_division=0)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6.5),
                                 gridspec_kw={"width_ratios": [n, 1.4]},
                                 sharey=True)
    else:
        axes = ax
    sns.heatmap(cm_norm, annot=False, cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor=PAPER, ax=axes[0],
                vmin=0, vmax=1)
    cmax = cm_norm.max()
    for i in range(n):
        for j in range(n):
            cell = cm_norm[i, j]
            txt = PAPER if cell > 0.55 * cmax else INK
            axes[0].text(j + 0.5, i + 0.5, f"{cell:.2f}",
                         ha="center", va="center", fontsize=10,
                         fontweight="bold", color=txt)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(title, color=INK)
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    # F1 strip
    f1_grid = f.reshape(-1, 1)
    sns.heatmap(f1_grid, annot=False, cmap="RdYlGn", cbar=False,
                xticklabels=["F1"], yticklabels=False,
                linewidths=0.5, linecolor=PAPER, ax=axes[1],
                vmin=0, vmax=1)
    for i in range(n):
        col = PAPER if f[i] < 0.4 or f[i] > 0.7 else INK
        axes[1].text(0.5, i + 0.5, f"{f[i]:.2f}",
                     ha="center", va="center", fontsize=11,
                     fontweight="bold", color=col)
    axes[1].set_title("Per-class F1", color=INK)
    plt.setp(axes[1].get_xticklabels(), rotation=0)
    plt.tight_layout()
    return axes


def per_class_recall(y_true, y_pred, class_names, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    rec = cm.diagonal() / cm.sum(axis=1)
    order = np.argsort(rec)[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([class_names[i] for i in order], rec[order],
                  color=[PALETTE[i] for i in order],
                  edgecolor=INK, lw=0.5, alpha=0.92)
    for b, r in zip(bars, rec[order]):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.015,
                f"{r:.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class recall (test set), sorted high to low",
                 color=INK)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ax


# ------------------------------------------------------------------
# Loss / learning curves
# ------------------------------------------------------------------
def loss_curve(history, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history],
            "-o", markersize=4, label="Train", color=PRIMARY, lw=1.6)
    if "val_loss" in history[0]:
        ax.plot(epochs, [h["val_loss"] for h in history],
                "-o", markersize=4, label="Validation", color=WARN, lw=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title, color=INK)
    ax.legend(loc="best")
    return ax


def learning_curve(sizes, train_scores, val_scores, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, train_scores, "-o", label="Train accuracy",
            color=PRIMARY, lw=2.0, ms=6)
    ax.plot(sizes, val_scores, "-o", label="Validation accuracy",
            color=WARN, lw=2.0, ms=6)
    ax.fill_between(sizes, train_scores, val_scores,
                    alpha=0.15, color=MUTED, label="Generalisation gap")
    final_gap = train_scores[-1] - val_scores[-1]
    ax.text(sizes[-1], (train_scores[-1] + val_scores[-1]) / 2,
            f"  gap={final_gap:.2f}", color=MUTED, fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, color=INK)
    ax.legend(loc="lower right")
    return ax


# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------
def feature_importance(names, importances, top=15, ax=None,
                       title_suffix="(random forest)"):
    order = np.argsort(importances)[::-1][:top]
    names_sorted = np.array(names)[order][::-1]
    vals = importances[order][::-1]

    def cat_color(name):
        if name.startswith("wilderness"):
            return "#5E548E"
        if name.startswith("soil"):
            return GOOD
        return PRIMARY

    colors = [cat_color(n) for n in names_sorted]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, top * 0.32)))
    bars = ax.barh(range(top), vals, color=colors,
                   edgecolor=INK, lw=0.4, alpha=0.92)
    ax.set_yticks(range(top))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top} feature importance {title_suffix}",
                 color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=PRIMARY, label="Continuous"),
        Patch(color="#5E548E", label="Wilderness"),
        Patch(color=GOOD, label="Soil type"),
    ], loc="lower right")
    return ax


def feature_importance_by_class(rf, X, y, feat_names, class_names,
                                minority_classes, top=5):
    """1-vs-rest gini importance from a small RF per minority class.

    Lighter than permutation importance — single RF.fit per class with a
    capped sample. Still reveals the per-class signature features."""
    from sklearn.ensemble import RandomForestClassifier
    n = len(minority_classes)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, max(3.5, top * 0.5)),
                             sharex=True)
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    sample_n = min(15000, len(y))
    idx = rng.choice(len(y), sample_n, replace=False)
    Xs, ys = X[idx], y[idx]
    for ax, cls in zip(axes, minority_classes):
        y_bin = (ys == cls).astype(int)
        clf = RandomForestClassifier(n_estimators=40, max_depth=8,
                                     n_jobs=-1, random_state=0,
                                     class_weight="balanced")
        clf.fit(Xs, y_bin)
        imean = clf.feature_importances_
        order = np.argsort(imean)[-top:]
        names_sorted = np.array(feat_names)[order]
        col = PALETTE[cls - 1] if y.min() >= 1 else PALETTE[cls]
        ax.barh(range(top), imean[order], color=col, edgecolor=INK,
                lw=0.4, alpha=0.92)
        ax.set_yticks(range(top))
        ax.set_yticklabels(names_sorted, fontsize=9)
        cls_idx = (cls - 1) if y.min() >= 1 else cls
        ax.set_title(class_names[cls_idx], color=INK, fontsize=10.5)
        ax.set_xlabel("Gini importance")
    fig.suptitle("Per-class top features: which signals actually separate each minority class",
                 fontsize=12, color=INK, y=1.02)
    plt.tight_layout()
    return axes


# ------------------------------------------------------------------
# PCA / KDE
# ------------------------------------------------------------------
def pca_scatter(X2, y, class_names, title, ax=None, s=4):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
    for c in np.unique(y):
        m = y == c
        ax.scatter(X2[m, 0], X2[m, 1], s=s, alpha=0.45,
                   label=class_names[c - 1] if y.min() >= 1 else class_names[c],
                   color=PALETTE[c - 1] if y.min() >= 1 else PALETTE[c],
                   edgecolor="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, color=INK)
    ax.legend(markerscale=4, fontsize=9, loc="best", framealpha=0.92)
    return ax


def elevation_kde(X, y, feat_names, class_names, ax=None):
    """1-D KDE of elevation by cover type — single chart that explains
    why elevation dominates RF importance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.8))
    elev_idx = feat_names.index("elevation") if "elevation" in feat_names else 0
    classes = np.unique(y)
    elev_min = X[:, elev_idx].min()
    elev_max = X[:, elev_idx].max()
    grid = np.linspace(elev_min, elev_max, 300)
    from scipy.stats import gaussian_kde
    for c, color in zip(classes, PALETTE):
        rng = np.random.default_rng(int(c))
        sub = X[y == c, elev_idx]
        if len(sub) > 4000:
            sub = rng.choice(sub, 4000, replace=False)
        kde = gaussian_kde(sub)
        d = kde(grid)
        ax.fill_between(grid, 0, d, alpha=0.30, color=color)
        ax.plot(grid, d, color=color, lw=1.6,
                label=class_names[c - 1] if y.min() >= 1 else class_names[c])
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Density")
    ax.set_title("Elevation alone separates most species: this is why RF feature-importance is dominated by elevation",
                 color=INK)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    return ax


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------
def model_comparison(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    df = df.sort_values("accuracy")
    y = np.arange(len(df))
    h = 0.4
    ax.barh(y - h / 2, df["accuracy"], height=h, label="Accuracy",
            color=PRIMARY, edgecolor=INK, lw=0.4, alpha=0.92)
    ax.barh(y + h / 2, df["macro F1"], height=h, label="Macro F1",
            color=ACCENT, edgecolor=INK, lw=0.4, alpha=0.92)
    for i, (a, f) in enumerate(zip(df["accuracy"], df["macro F1"])):
        ax.text(a + 0.005, i - h / 2, f"{a:.2f}", va="center", fontsize=9)
        ax.text(f + 0.005, i + h / 2, f"{f:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.08)
    ax.set_title("Test-set scores by model: tree ensembles dominate the linear/NN floor",
                 color=INK)
    ax.legend(loc="lower right")
    return ax


# ------------------------------------------------------------------
# Ridgeline (joyplot) — feature partition by class
# ------------------------------------------------------------------
def ridgeline(x, y, class_names, feat_label="Elevation (m)",
              title="", ax=None, overlap=1.15):
    """Stacked per-class densities, ordered by median.

    The most legible way to show a feature carving classes apart: read
    straight off which band of the axis belongs to which class. `x` is the
    raw feature (e.g. metres, not standardised) so the axis stays physical."""
    from scipy.stats import gaussian_kde
    classes = np.unique(y)
    order = sorted(classes, key=lambda c: np.median(x[y == c]))
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 0.78 * len(order) + 1.6))
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    grid = np.linspace(lo, hi, 400)
    rng = np.random.default_rng(0)
    dens = {}
    for c in order:
        sub = x[y == c]
        if len(sub) > 6000:
            sub = rng.choice(sub, 6000, replace=False)
        dens[c] = gaussian_kde(sub)(grid)
    peak = max(d.max() for d in dens.values())
    scale = overlap / peak
    for i, c in enumerate(order):
        base = i
        d = dens[c] * scale
        color = PALETTE[c] if c < len(PALETTE) else MUTED
        ax.fill_between(grid, base, base + d, color=color, alpha=0.82,
                        zorder=len(order) - i, lw=0)
        ax.plot(grid, base + d, color=PAPER, lw=0.9, zorder=len(order) - i)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([class_names[c] for c in order])
    ax.set_ylim(-0.3, len(order) + overlap)
    ax.set_xlabel(feat_label)
    ax.spines["left"].set_visible(False)
    ax.grid(False, axis="y")
    if title:
        ax.set_title(title, color=INK)
    return ax


# ------------------------------------------------------------------
# Spatial CV — random split vs leave-one-region-out
# ------------------------------------------------------------------
def cv_gap_dumbbell(metrics, title="", ax=None):
    """Dumbbell: each metric's random-split value vs its leave-one-region-out
    value, with the optimism gap labelled between them. `metrics` is a list of
    (name, random_value, spatial_value)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 1.4 + 0.95 * len(metrics)))
    for i, (name, r, s) in enumerate(metrics):
        ax.plot([s, r], [i, i], color="#C7CDD4", lw=2.5, zorder=1)
        ax.scatter([r], [i], color=PRIMARY, s=130, zorder=3,
                   label="Random 70/15/15 split" if i == 0 else None,
                   edgecolor=PAPER, lw=1.2)
        ax.scatter([s], [i], color=WARN, s=130, zorder=3,
                   label="Leave-one-region-out" if i == 0 else None,
                   edgecolor=PAPER, lw=1.2)
        ax.annotate(f"{r:.2f}", (r, i), textcoords="offset points",
                    xytext=(8, 9), fontsize=9.5, color=PRIMARY, fontweight="bold")
        ax.annotate(f"{s:.2f}", (s, i), textcoords="offset points",
                    xytext=(-8, 9), fontsize=9.5, color=WARN,
                    fontweight="bold", ha="right")
        ax.annotate(f"optimism −{r - s:.2f}", ((r + s) / 2, i),
                    textcoords="offset points", xytext=(0, -16),
                    fontsize=8.5, color=MUTED, ha="center", style="italic")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.legend(loc="lower left", ncol=2)
    if title:
        ax.set_title(title, color=INK)
    return ax


# ------------------------------------------------------------------
# Calibration / reliability
# ------------------------------------------------------------------
def reliability_diagram(y_true, proba, title="Reliability", ax=None, n_bins=12):
    """Reliability curve with the calibration gap shaded and ECE annotated.
    A faint confidence histogram along the bottom shows where the mass lives."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    mean_conf = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if m.sum():
            acc[b] = correct[m].mean()
            mean_conf[b] = conf[m].mean()
    ece = np.nansum(cnt * np.abs(acc - mean_conf)) / cnt.sum()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.3,
            label="Perfect calibration")
    good = ~np.isnan(acc)
    # shaded gap between observed accuracy and the diagonal
    ax.fill_between(centers[good], acc[good], centers[good],
                    color=WARN, alpha=0.18, label="Calibration gap")
    ax.plot(centers[good], acc[good], "-o", color=PRIMARY, lw=2,
            ms=6, label="Observed accuracy")
    # confidence histogram along the bottom (scaled into [0, 0.18])
    h = cnt / cnt.max() * 0.16
    ax.bar(centers, h, width=1 / n_bins * 0.9, bottom=0,
           color=MUTED, alpha=0.25, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"{title}   (ECE = {ece:.3f})", color=INK)
    ax.legend(loc="upper left")
    return ax, ece


# ------------------------------------------------------------------
# Bootstrap-CI forest plot for a metric
# ------------------------------------------------------------------
def metric_forest(rows, title="", xlabel="Macro-F1 (95% bootstrap CI)", ax=None):
    """Caterpillar/forest plot: (label, point, lo, hi) per model, best on top."""
    rows = sorted(rows, key=lambda r: r[1])
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.2))
    for i, (label, p, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color="#C7CDD4", lw=3,
                solid_capstyle="round", zorder=1)
        ax.scatter([p], [i], color=PRIMARY, s=95, zorder=3,
                   edgecolor=PAPER, lw=1.0)
        ax.text(hi + 0.006, i, f"{p:.3f}  [{lo:.3f}, {hi:.3f}]",
                va="center", fontsize=8.8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_xlim(right=min(1.0, max(r[3] for r in rows) + 0.16))
    if title:
        ax.set_title(title, color=INK)
    return ax


# ------------------------------------------------------------------
# Predictive uncertainty
# ------------------------------------------------------------------
def entropy_by_correctness(y_true, proba, title="", ax=None):
    """Predictive-entropy density split by whether the prediction was right.
    If errors sit at visibly higher entropy, the model's own uncertainty is a
    usable triage signal — route the high-entropy tail to a human reviewer."""
    from scipy.stats import gaussian_kde
    eps = 1e-12
    ent = -(proba * np.log(proba + eps)).sum(axis=1)
    pred = proba.argmax(axis=1)
    correct = pred == y_true
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
    grid = np.linspace(0, max(ent.max(), 1e-3), 300)
    for mask, color, label in [(correct, GOOD, "Correct predictions"),
                               (~correct, WARN, "Errors")]:
        sub = ent[mask]
        if len(sub) < 5:
            continue
        if len(sub) > 8000:
            sub = np.random.default_rng(0).choice(sub, 8000, replace=False)
        d = gaussian_kde(sub)(grid)
        ax.fill_between(grid, 0, d, color=color, alpha=0.30)
        ax.plot(grid, d, color=color, lw=1.9, label=label)
    med_c = np.median(ent[correct])
    med_e = np.median(ent[~correct])
    ax.axvline(med_c, color=GOOD, ls=":", lw=1.2)
    ax.axvline(med_e, color=WARN, ls=":", lw=1.2)
    ax.set_xlabel("Predictive entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title(title or "Errors carry higher entropy — uncertainty is a usable triage signal",
                 color=INK)
    ax.legend(loc="upper right")
    return ax, (med_c, med_e)
