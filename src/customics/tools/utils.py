"""Data utilities: sample alignment, splitting, and visualisation helpers."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.manifold import TSNE


sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


# ---------------------------------------------------------------------------
# Sample alignment
# ---------------------------------------------------------------------------


def get_common_samples(dfs: List[pd.DataFrame]) -> List[str]:
    """Return sample IDs present in every DataFrame.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        DataFrames whose indices are sample IDs.

    Returns
    -------
    list of str
        Sorted list of common sample IDs.
    """
    common = set(dfs[0].index)
    for df in dfs[1:]:
        common &= set(df.index)
    return sorted(common)


def get_sub_omics_df(
    omics_df: Dict[str, pd.DataFrame], lt_samples: List[str]
) -> Dict[str, pd.DataFrame]:
    """Subset every omics DataFrame to the given samples.

    Parameters
    ----------
    omics_df : dict
        Multi-omics dictionary (source name → DataFrame).
    lt_samples : list of str
        Sample IDs to keep.

    Returns
    -------
    dict
        Same structure as ``omics_df`` but rows restricted to ``lt_samples``.
    """
    return {key: df.loc[lt_samples, :] for key, df in omics_df.items()}


# ---------------------------------------------------------------------------
# Cross-validation split management
# ---------------------------------------------------------------------------


def save_splits(lt_samples: List[str], cohort: str, split_dir: str = "splits") -> None:
    """Compute 5-fold cross-validation splits and persist them to disk.

    Parameters
    ----------
    lt_samples : list of str
        All sample IDs.
    cohort : str
        Cohort name used to create a subdirectory under ``split_dir``.
    split_dir : str
        Root directory for split files.
    """
    kf = KFold(n_splits=5)
    out_dir = os.path.join(split_dir, cohort)
    os.makedirs(out_dir, exist_ok=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(lt_samples), start=1):
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15)
        for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            with open(os.path.join(out_dir, f"split_{name}_{i}.txt"), "w") as f:
                for idx in indices:
                    f.write(lt_samples[idx] + "\n")


def get_splits(
    cohort: str, split: int, split_dir: str = "splits"
) -> tuple[List[str], List[str], List[str]]:
    """Load pre-computed train/val/test sample IDs for a given fold.

    Parameters
    ----------
    cohort : str
        Cohort name.
    split : int
        Fold index (1-based).
    split_dir : str
        Root directory containing split files.

    Returns
    -------
    tuple of (list, list, list)
        ``(samples_train, samples_val, samples_test)``.
    """
    out_dir = os.path.join(split_dir, cohort)
    result = []
    for name in ("train", "val", "test"):
        path = os.path.join(out_dir, f"split_{name}_{split}.txt")
        with open(path) as f:
            result.append([line.rstrip() for line in f])
    return tuple(result)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def save_plot_score(
    filename: str,
    z: np.ndarray,
    y: np.ndarray,
    title: str,
    show: bool = False,
) -> None:
    """Compute a t-SNE embedding and save a colour-coded scatter plot.

    Parameters
    ----------
    filename : str
        Output file path (without extension; a ``.png`` suffix is appended).
    z : np.ndarray
        High-dimensional feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Class labels for colouring, shape (n_samples,).
    title : str
        Plot title.
    show : bool
        If True, display the plot interactively after saving.
    """
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    embedding = tsne.fit_transform(z)
    df = pd.DataFrame(
        {"targets": y, "x-axis": embedding[:, 0], "y-axis": embedding[:, 1]}
    )
    sns.scatterplot(
        x="x-axis",
        y="y-axis",
        hue=df["targets"].tolist(),
        palette=sns.color_palette("hls", len(np.unique(y))),
        data=df,
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.5, 1.1), loc=2, borderaxespad=0.0)
    plt.savefig(filename + ".png", bbox_inches="tight")
    if show:
        plt.show()
    plt.clf()
