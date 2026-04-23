"""Standard (deterministic) encoder network."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn

from customics.tools.net_utils import FullyConnectedLayer


class Encoder(nn.Module):
    """Deterministic encoder that maps high-dimensional input to a latent vector.

    Architecture: ``InputLayer → [HiddenLayers...] → OutputLayer``
    where each hidden layer is a :class:`~customics.tools.net_utils.FullyConnectedLayer`.

    Parameters
    ----------
    input_dim : int
        Dimension of the input tensor.
    hidden_dim : list of int
        Sizes of intermediate hidden layers.
    latent_dim : int
        Dimension of the output latent representation.
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
        layers["OutputLayer"] = FullyConnectedLayer(
            hidden_dim[-1], latent_dim,
            norm_layer=norm_layer, leaky_slope=leaky_slope,
            dropout=0.0, activation=False, normalization=False,
        )
        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` to a latent vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Latent tensor, shape (batch, latent_dim).
        """
        return self.net(x)
