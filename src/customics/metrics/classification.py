"""Classification evaluation metrics and plots."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder


def roc_auc_score_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ohe: OneHotEncoder,
    average: str = "macro",
) -> float:
    """Compute multi-class ROC-AUC using one-vs-one strategy.

    Parameters
    ----------
    y_true : np.ndarray
        Integer class labels.
    y_pred : np.ndarray
        Predicted class probabilities, shape (n_samples, n_classes).
    ohe : OneHotEncoder
        Fitted encoder used to binarise ``y_true``.
    average : str
        Averaging strategy passed to :func:`sklearn.metrics.roc_auc_score`.

    Returns
    -------
    float
        ROC-AUC score.
    """
    y_true_bin = ohe.transform(np.array(y_true).reshape(-1, 1))
    return roc_auc_score(y_true_bin, y_pred, average=average, multi_class="ovo")


def multi_classification_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    average: str = "weighted",
    save_confusion: bool = False,
    filename: Optional[str] = None,
    ohe: Optional[OneHotEncoder] = None,
) -> Dict[str, float]:
    """Compute a standard classification metrics dictionary.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    y_pred_proba : np.ndarray
        Predicted class probabilities, shape (n_samples, n_classes).
    average : str
        Averaging strategy for precision, recall, and F1.
    save_confusion : bool
        If True, save a confusion matrix heatmap to ``filename``.
    filename : str, optional
        Path prefix for the confusion matrix image (without extension).
    ohe : OneHotEncoder, optional
        Required when computing AUC.

    Returns
    -------
    dict
        Keys: ``'Accuracy'``, ``'F1-score'``, ``'Precision'``, ``'Recall'``,
        ``'AUC'``.
    """
    scores: Dict[str, float] = {
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "F1-score": metrics.f1_score(y_true, y_pred, average=average),
        "Precision": metrics.precision_score(y_true, y_pred, average=average),
        "Recall": metrics.recall_score(y_true, y_pred, average=average),
        "AUC": roc_auc_score_multiclass(y_true, y_pred_proba, ohe, average=average)
        if ohe is not None
        else float("nan"),
    }
    if save_confusion and filename is not None:
        plt.figure(figsize=(18, 8))
        sns.heatmap(
            metrics.confusion_matrix(y_true, y_pred),
            annot=True,
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true),
            cmap="summer",
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(filename + ".png")
        plt.clf()
    return scores


def plot_roc_multiclass(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    filename: str = "",
    n_classes: int = 2,
    var_names: Optional[List[str]] = None,
) -> None:
    """Plot per-class ROC curves for a multi-class model.

    Parameters
    ----------
    y_test : np.ndarray
        Ground-truth integer labels.
    y_pred_proba : np.ndarray
        Predicted class probabilities, shape (n_samples, n_classes).
    filename : str
        If non-empty, save the figure to ``roc_multi_{filename}.png``.
    n_classes : int
        Number of classes.
    var_names : list of str, optional
        Class names used in the legend.
    """
    if var_names is None:
        var_names = [str(i) for i in range(n_classes)]

    colors = ["red", "green", "blue", "magenta", "orange", "cyan"]
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=1, label=f"{var_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve — {filename}")
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(f"roc_multi_{filename}.png")
