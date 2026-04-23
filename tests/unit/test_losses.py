"""Unit tests for loss functions."""

import numpy as np
import pytest
import torch

from customics.loss.classification_loss import classification_loss
from customics.loss.mmd_loss import compute_kernel, compute_mmd
from customics.loss.survival_loss import CoxLoss


class TestClassificationLoss:
    def test_ce_returns_scalar(self):
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        loss = classification_loss("CE", logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_bce_returns_scalar(self):
        logits = torch.randn(8, 1)
        labels = torch.randint(0, 2, (8, 1)).float()
        loss = classification_loss("BCE", logits, labels)
        assert loss.ndim == 0

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            classification_loss("MSE", torch.randn(4, 2), torch.zeros(4, dtype=torch.long))

    def test_gradients_flow(self):
        logits = torch.randn(8, 3, requires_grad=True)
        labels = torch.randint(0, 3, (8,))
        loss = classification_loss("CE", logits, labels)
        loss.backward()
        assert logits.grad is not None


class TestMMDLoss:
    def test_kernel_shape(self):
        x = torch.randn(5, 8)
        y = torch.randn(7, 8)
        k = compute_kernel(x, y)
        assert k.shape == (5, 7)

    def test_mmd_self_zero(self):
        x = torch.randn(16, 8)
        mmd = compute_mmd(x, x)
        assert abs(mmd.item()) < 1e-5

    def test_mmd_different_distributions(self):
        x = torch.zeros(32, 8)
        y = torch.ones(32, 8) * 10
        mmd = compute_mmd(x, y)
        assert mmd.item() > 0


class TestCoxLoss:
    def test_returns_scalar(self):
        n = 10
        survtime = torch.tensor(np.random.randint(100, 2000, n), dtype=torch.float32)
        censor = torch.tensor(np.random.randint(0, 2, n), dtype=torch.float32)
        hazard = torch.randn(n, 1)
        loss = CoxLoss(survtime, censor, hazard, torch.device("cpu"))
        assert loss.ndim == 0

    def test_gradients_flow(self):
        n = 8
        survtime = torch.arange(1, n + 1, dtype=torch.float32)
        censor = torch.ones(n)
        hazard = torch.randn(n, 1, requires_grad=True)
        loss = CoxLoss(survtime, censor, hazard, torch.device("cpu"))
        loss.backward()
        assert hazard.grad is not None
