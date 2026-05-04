"""Plot helpers used by the notebook."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def confusion(y_true, y_pred, class_names, title, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return ax


def loss_curve(history, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    if "val_loss" in history[0]:
        ax.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title(title)
    ax.legend()
    return ax


def learning_curve(sizes, train_scores, val_scores, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, train_scores, "o-", label="train")
    ax.plot(sizes, val_scores, "o-", label="val")
    ax.set_xlabel("training set size")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    return ax


def feature_importance(names, importances, top=15, ax=None):
    order = np.argsort(importances)[::-1][:top]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(top), importances[order][::-1])
    ax.set_yticks(range(top))
    ax.set_yticklabels(np.array(names)[order][::-1])
    ax.set_xlabel("importance")
    return ax


def pca_scatter(X2, y, class_names, title, ax=None, s=2):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    for c in np.unique(y):
        m = y == c
        ax.scatter(X2[m, 0], X2[m, 1], s=s, alpha=0.4, label=class_names[c])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    return ax
