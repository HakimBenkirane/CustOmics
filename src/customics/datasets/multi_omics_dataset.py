"""PyTorch Dataset classes for multi-omics data."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiOmicsDataset(Dataset):
    """A PyTorch Dataset that pairs multi-omics matrices with clinical outcomes.

    Each sample returns a list of per-source tensors together with its class
    label, survival time, and event indicator.

    Parameters
    ----------
    omics_df : dict of str → pd.DataFrame
        Multi-omics data.  Each DataFrame must be indexed by sample ID.
    clinical_df : pd.DataFrame
        Clinical metadata indexed by sample ID.
    lt_samples : list of str
        Ordered list of sample IDs to include (must be present in every
        omics DataFrame and in ``clinical_df``).
    label : str or None
        Column in ``clinical_df`` containing class labels.  Pass ``None`` to
        disable label loading (returns 0).
    event : str
        Column in ``clinical_df`` containing the event indicator (0/1).
    surv_time : str
        Column in ``clinical_df`` containing survival time.

    Examples
    --------
    >>> dataset = MultiOmicsDataset(omics_df, clinical_df, samples, "PAM50", "OS", "OS.time")
    >>> omics_tensors, label, time, event = dataset[0]
    """

    def __init__(
        self,
        omics_df: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        lt_samples: List[str],
        label: Optional[str],
        event: str,
        surv_time: str,
    ) -> None:
        self.omics_df = omics_df
        self.clinical_df = clinical_df
        self.lt_samples = lt_samples
        self.label = label
        self.event = event
        self.surv_time = surv_time

    def __len__(self) -> int:
        return len(self.lt_samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[List[torch.Tensor], int, int, int]:
        sample = self.lt_samples[index]
        omics_data = [
            torch.tensor(df.loc[sample, :].values.astype(np.float32))
            for df in self.omics_df.values()
        ]
        lbl = self.clinical_df.loc[sample, self.label] if self.label else 0
        os_time = int(self.clinical_df.loc[sample, self.surv_time])
        os_event = int(self.clinical_df.loc[sample, self.event])
        return omics_data, lbl, os_time, os_event

    def get_samples(self) -> List[str]:
        """Return the list of sample IDs in dataset order."""
        return self.lt_samples
