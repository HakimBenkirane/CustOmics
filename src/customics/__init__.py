"""
customics — A versatile deep-learning based strategy for multi-omics integration.

Quick start
-----------
>>> import torch
>>> from customics import CustOMICS
>>> model = CustOMICS(
...     source_params={...},
...     central_params={...},
...     classif_params={...},
...     surv_params={...},
...     train_params={"switch": 5, "lr": 1e-3},
...     device=torch.device("cpu"),
... )
>>> model.fit(omics_train, clinical_df, label="PAM50", event="OS", surv_time="OS.time")
>>> metrics = model.evaluate(omics_test, clinical_df, label="PAM50",
...                          event="OS", surv_time="OS.time", task="classification")
"""

from customics.network.customics import CustOMICS
from customics.datasets.multi_omics_dataset import MultiOmicsDataset
from customics.metrics.classification import multi_classification_evaluation
from customics.metrics.survival import CIndex_lifeline
from customics.tools.utils import get_common_samples, get_sub_omics_df
from customics.exceptions import (
    CustOmicsError,
    DataValidationError,
    ModelNotFittedError,
    ConfigurationError,
)

__version__ = "0.1.0"
__author__ = "Hakim Benkirane"
__email__ = "hakim.benkirane@centralesupelec.fr"

__all__ = [
    # Core model
    "CustOMICS",
    # Dataset
    "MultiOmicsDataset",
    # Metrics
    "multi_classification_evaluation",
    "CIndex_lifeline",
    # Data utilities
    "get_common_samples",
    "get_sub_omics_df",
    # Exceptions
    "CustOmicsError",
    "DataValidationError",
    "ModelNotFittedError",
    "ConfigurationError",
]
