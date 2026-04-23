"""Maximum Mean Discrepancy (MMD) loss for VAE regularisation."""

from __future__ import annotations

import torch


def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the RBF kernel matrix between two sample sets.

    Parameters
    ----------
    x : torch.Tensor
        Shape (n, d).
    y : torch.Tensor
        Shape (m, d).

    Returns
    -------
    torch.Tensor
        Kernel matrix of shape (n, m).
    """
    x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    return torch.exp(-(tiled_x - tiled_y).pow(2).mean(2) / float(dim))


def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Maximum Mean Discrepancy between distributions ``x`` and ``y``.

    Parameters
    ----------
    x : torch.Tensor
        Samples from the first distribution, shape (n, d).
    y : torch.Tensor
        Samples from the second distribution, shape (m, d).

    Returns
    -------
    torch.Tensor
        Scalar MMD value.
    """
    return compute_kernel(x, x).mean() + compute_kernel(y, y).mean() - 2 * compute_kernel(x, y).mean()
