"""Survival analysis loss functions (Cox partial likelihood)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def CoxLoss(
    survtime: torch.Tensor,
    censor: torch.Tensor,
    hazard_pred: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute the Cox partial log-likelihood loss (cox-nnet formulation).

    Credit: Travers Ching — cox-nnet (https://github.com/traversc/cox-nnet).

    Parameters
    ----------
    survtime : torch.Tensor
        Observed survival times, shape (batch,).
    censor : torch.Tensor
        Event indicator (1 = event, 0 = censored), shape (batch,).
    hazard_pred : torch.Tensor
        Predicted log-hazard scores, shape (batch, 1) or (batch,).
    device : torch.device
        Device on which to place intermediate tensors.

    Returns
    -------
    torch.Tensor
        Scalar Cox loss.
    """
    n = len(survtime)
    R_mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            R_mat[i, j] = int(survtime[j] >= survtime[i])
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)


class NegativeLogLikelihood(nn.Module):
    """DeepSurv-style negative log-likelihood loss with L2 regularisation.

    Parameters
    ----------
    l2_reg : float
        L2 regularisation weight applied to model weights.
    """

    def __init__(self, l2_reg: float) -> None:
        super().__init__()
        self.l2_reg = l2_reg

    def forward(
        self,
        risk_pred: torch.Tensor,
        y: torch.Tensor,
        e: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        l2_loss = sum(
            torch.norm(w, p=2)
            for name, w in model.named_parameters()
            if "weight" in name
        )
        return neg_log_loss + self.l2_reg * l2_loss
