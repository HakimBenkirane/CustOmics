"""Probabilistic (VAE) encoder network."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from customics.tools.net_utils import FullyConnectedLayer


class ProbabilisticEncoder(nn.Module):
    """Inference network for the variational autoencoder.

    Outputs the mean and log-variance of the approximate posterior
    ``q(z | x)``.

    Parameters
    ----------
    input_dim : int
        Dimension of the input tensor.
    hidden_dim : list of int
        Sizes of intermediate hidden layers.
    latent_dim : int
        Dimension of the latent space.
    norm_layer : type or bool
        Normalization layer class or ``True`` for ``nn.BatchNorm1d``.
    leaky_slope : float
        Negative slope for LeakyReLU activations.
    dropout : float
        Dropout rate (0 = disabled).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: List[int],
        latent_dim: int,
        norm_layer: Union[type, bool] = nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["InputLayer"] = FullyConnectedLayer(
            input_dim, hidden_dim[0],
            norm_layer=norm_layer, leaky_slope=leaky_slope,
            dropout=dropout, activation=True,
        )
        for i in range(1, len(hidden_dim)):
            layers[f"Layer{i}"] = FullyConnectedLayer(
                hidden_dim[i - 1], hidden_dim[i],
                norm_layer=norm_layer, leaky_slope=leaky_slope,
                dropout=dropout if i % 2 == 0 else 0.0, activation=True,
            )
        self.net = nn.Sequential(layers)
        self.mean_layer = FullyConnectedLayer(
            hidden_dim[-1], latent_dim,
            norm_layer=norm_layer, leaky_slope=leaky_slope,
            dropout=0.0, activation=False, normalization=False,
        )
        self.log_var_layer = FullyConnectedLayer(
            hidden_dim[-1], latent_dim,
            norm_layer=norm_layer, leaky_slope=leaky_slope,
            dropout=0.0, activation=False, normalization=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior mean and log-variance.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim).

        Returns
        -------
        mean : torch.Tensor
            Posterior mean, shape (batch, latent_dim).
        log_var : torch.Tensor
            Posterior log-variance, shape (batch, latent_dim).
        """
        h = self.net(x)
        return self.mean_layer(h), self.log_var_layer(h)
