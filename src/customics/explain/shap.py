"""SHAP-based explainability utilities for customics models."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from customics.network.customics import CustOMICS


class ModelWrapper(nn.Module):
    """Thin wrapper that exposes a single-source forward pass for SHAP.

    The wrapper takes a raw omics tensor for one source and returns class
    logits, making it compatible with :class:`shap.DeepExplainer`.

    Parameters
    ----------
    model : CustOMICS
        A fitted customics model.
    source : str
        The omics source key to explain.
    """

    def __init__(self, model: "CustOMICS", source: str) -> None:
        super().__init__()
        self.model = model
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.source_predict(x, self.source)


def processPhenotypeDataForSamples(
    clinical_df: pd.DataFrame,
    sample_id: List[str],
    label_encoder,
) -> pd.DataFrame:
    """Subset clinical DataFrame to the given samples.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Full clinical metadata.
    sample_id : list of str
        Sample IDs to retain.
    label_encoder : LabelEncoder
        Fitted label encoder (unused here; kept for API compatibility).

    Returns
    -------
    pd.DataFrame
        Subsetted clinical DataFrame.
    """
    return clinical_df.loc[sample_id, :]


def randomTrainingSample(expr: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Draw a random subset of rows from an expression DataFrame.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix, rows are samples.
    sample_size : int
        Number of rows to sample.

    Returns
    -------
    pd.DataFrame
        Randomly sampled rows.
    """
    return expr.sample(n=sample_size, axis=0)


def splitExprandSample(
    condition: pd.Series,
    sample_size: int,
    expr: pd.DataFrame,
) -> pd.DataFrame:
    """Filter rows by a boolean condition and draw a random subset.

    Parameters
    ----------
    condition : pd.Series
        Boolean mask aligned with ``expr`` rows.
    sample_size : int
        Number of rows to sample from the filtered set.
    expr : pd.DataFrame
        Expression matrix.

    Returns
    -------
    pd.DataFrame
        Filtered and sampled expression rows.
    """
    return expr[condition].sample(n=sample_size, axis=0)


def addToTensor(expr_selection: pd.DataFrame, device: str) -> torch.Tensor:
    """Convert an expression DataFrame to a float32 tensor on ``device``.

    Parameters
    ----------
    expr_selection : pd.DataFrame
        Expression matrix to convert.
    device : str
        Target device string (e.g. ``'cpu'`` or ``'cuda'``).

    Returns
    -------
    torch.Tensor
        Float32 tensor on the specified device.
    """
    return torch.tensor(
        expr_selection.values.astype("float32"), dtype=torch.float32
    ).to(device)
