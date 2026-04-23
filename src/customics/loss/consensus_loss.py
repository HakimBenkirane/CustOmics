"""Cross-modal consensus reconstruction loss."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def consensus_loss(x: List[torch.Tensor], autoencoders: nn.ModuleList) -> torch.Tensor:
    """Compute the cross-modal consensus loss.

    For every pair of sources (i, j), measures how well source i's encoder
    followed by source j's decoder can reconstruct source j's input.

    Parameters
    ----------
    x : list of torch.Tensor
        Per-source input tensors.
    autoencoders : nn.ModuleList
        Per-source autoencoders with accessible ``encoder`` and ``decoder``
        sub-modules.

    Returns
    -------
    torch.Tensor
        Scalar consensus loss.
    """
    mse = nn.MSELoss()
    loss = torch.tensor(0.0, device=x[0].device)
    for i, ae_i in enumerate(autoencoders):
        for j, ae_j in enumerate(autoencoders):
            rep_i = ae_i.encoder(x[i])
            recon_j = ae_j.decoder(rep_i)
            loss = loss + mse(recon_j, x[j])
    return loss
