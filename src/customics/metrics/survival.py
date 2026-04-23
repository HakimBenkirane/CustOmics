"""Survival analysis evaluation metrics."""

from __future__ import annotations

import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def CIndex_lifeline(
    hazards: np.ndarray,
    labels: np.ndarray,
    survtime_all: np.ndarray,
) -> float:
    """Compute the concordance index (C-index).

    Parameters
    ----------
    hazards : np.ndarray
        Predicted hazard scores, shape (n_samples, 1) or (n_samples,).
    labels : np.ndarray
        Event indicators (1 = event, 0 = censored).
    survtime_all : np.ndarray
        Observed survival times.

    Returns
    -------
    float
        Concordance index in [0, 1].
    """
    return concordance_index(survtime_all, -hazards, labels)


def cox_log_rank(
    hazardsdata: np.ndarray,
    labels: np.ndarray,
    survtime_all: np.ndarray,
) -> float:
    """Compute the log-rank test p-value after dichotomising hazard scores at the median.

    Parameters
    ----------
    hazardsdata : np.ndarray
        Predicted hazard scores, shape (n_samples,).
    labels : np.ndarray
        Event indicators.
    survtime_all : np.ndarray
        Observed survival times.

    Returns
    -------
    float
        Log-rank p-value.
    """
    median = np.median(hazardsdata)
    high_risk = hazardsdata > median
    results = logrank_test(
        survtime_all[~high_risk],
        survtime_all[high_risk],
        event_observed_A=labels[~high_risk],
        event_observed_B=labels[high_risk],
    )
    return results.p_value


def accuracy_cox(hazardsdata: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correctly stratified patients (median-split accuracy).

    Parameters
    ----------
    hazardsdata : np.ndarray
        Predicted hazard scores.
    labels : np.ndarray
        Ground-truth event indicators.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    predicted = (hazardsdata > np.median(hazardsdata)).astype(int)
    return np.mean(predicted == labels)
