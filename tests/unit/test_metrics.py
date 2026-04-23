"""Unit tests for evaluation metrics."""

import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from customics.metrics.classification import multi_classification_evaluation
from customics.metrics.survival import CIndex_lifeline, cox_log_rank, accuracy_cox


class TestClassificationMetrics:
    def setup_method(self):
        rng = np.random.default_rng(0)
        n, n_class = 40, 3
        self.y_true = rng.integers(0, n_class, n)
        self.y_pred = rng.integers(0, n_class, n)
        # Perfectly calibrated probabilities for each true class
        self.y_proba = rng.dirichlet(np.ones(n_class), size=n)
        self.ohe = OneHotEncoder(sparse_output=False).fit(
            self.y_true.reshape(-1, 1)
        )

    def test_returns_all_keys(self):
        scores = multi_classification_evaluation(
            self.y_true, self.y_pred, self.y_proba, ohe=self.ohe
        )
        for key in ("Accuracy", "F1-score", "Precision", "Recall", "AUC"):
            assert key in scores

    def test_perfect_accuracy(self):
        scores = multi_classification_evaluation(
            self.y_true, self.y_true, self.y_proba, ohe=self.ohe
        )
        assert scores["Accuracy"] == pytest.approx(1.0)

    def test_values_in_range(self):
        scores = multi_classification_evaluation(
            self.y_true, self.y_pred, self.y_proba, ohe=self.ohe
        )
        for key in ("Accuracy", "F1-score", "Precision", "Recall", "AUC"):
            assert 0.0 <= scores[key] <= 1.0


class TestSurvivalMetrics:
    def setup_method(self):
        rng = np.random.default_rng(1)
        n = 50
        self.survtime = rng.integers(100, 3000, n).astype(float)
        self.events = rng.integers(0, 2, n)
        self.hazards = rng.standard_normal((n, 1))

    def test_cindex_range(self):
        ci = CIndex_lifeline(self.hazards, self.events, self.survtime)
        assert 0.0 <= ci <= 1.0

    def test_perfect_cindex(self):
        # Higher survtime → lower hazard → perfect concordance
        perfect_hazard = -self.survtime.reshape(-1, 1).astype(float)
        ci = CIndex_lifeline(perfect_hazard, np.ones(len(self.survtime)), self.survtime)
        assert ci == pytest.approx(1.0)

    def test_cox_log_rank_returns_float(self):
        p = cox_log_rank(self.hazards.reshape(-1), self.events, self.survtime)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_accuracy_cox_range(self):
        acc = accuracy_cox(self.hazards.reshape(-1), self.events)
        assert 0.0 <= acc <= 1.0
