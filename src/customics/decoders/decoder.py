"""Standard (deterministic) decoder network."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn

from customics.tools.net_utils import FullyConnectedLayer


class Decoder(nn.Module):
    """Deterministic decoder that maps a latent vector back to data space.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation.
    hidden_dim : list of int
        Sizes of intermediate hidden layers (in encoder order; reversed
        internally).
    output_dim : int
        Dimension of the reconstructed output.
    norm_layer : type or bool
        Normalization layer class or ``True`` for ``nn.BatchNorm1d``.
    leaky_slope : float
        Negative slope for LeakyReLU activations.
    dropout : float
        Dropout rate (0 = disabled).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: List[int],
        output_dim: int,
        norm_layer: Union[type, bool] = nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rev = list(reversed(hidden_dim))
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["InputLayer"] = FullyConnectedLayer(
            latent_dim, rev[0],
            norm_layer=norm_layer, leaky_slope=leaky_slope,
            dropout=dropout, activation=True,
        )
        for i in range(1, len(rev)):
            layers[f"Layer{i}"] = FullyConnectedLayer(
                rev[i - 1], rev[i],
                norm_layer=norm_layer, leaky_slope=leaky_slope,
                dropout=dropout if i % 2 == 0 else 0.0, activation=True,
            )
        layers["OutputLayer"] = FullyConnectedLayer(
            rev[-1], output_dim,
            norm_layer=norm_layer, leaky_slope=leaky_slope,
            dropout=0.0, activation=False, normalization=False,
        )
        self.net = nn.Sequential(layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector ``z`` to data space.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor, shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor, shape (batch, output_dim).
        """
        return self.net(z)
