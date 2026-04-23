"""Survival prediction network (Cox proportional hazard head)."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from customics.exceptions import ConfigurationError


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "SELU": nn.SELU,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ELU": nn.ELU,
}


class SurvivalNet(nn.Module):
    """Fully-connected network that predicts log-hazard scores for Cox analysis.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:

        * ``dims`` (list of int): layer sizes including input and output.
        * ``drop`` (float): dropout probability.
        * ``norm`` (bool): whether to apply batch normalisation.
        * ``activation`` (str): activation function name (one of
          ``'SELU'``, ``'ReLU'``, ``'LeakyReLU'``, ``'Tanh'``,
          ``'Sigmoid'``, ``'ELU'``).

    Raises
    ------
    ConfigurationError
        If ``activation`` is not in the supported set.

    Examples
    --------
    >>> net = SurvivalNet({"dims": [128, 64, 32, 1], "drop": 0.2,
    ...                    "norm": True, "activation": "SELU"})
    >>> hazard = net(torch.randn(8, 128))   # shape (8, 1)
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.drop: float = config["drop"]
        self.norm: bool = config["norm"]
        self.dims: List[int] = config["dims"]
        self.activation: str = config["activation"]
        self.model = self._build_network()

    def _build_network(self) -> nn.Sequential:
        if self.activation not in _ACTIVATIONS:
            raise ConfigurationError(
                f"Unsupported activation '{self.activation}'. "
                f"Choose from {sorted(_ACTIVATIONS)}."
            )
        layers: list[nn.Module] = []
        for i in range(len(self.dims) - 1):
            if i > 0 and self.drop:
                layers.append(nn.Dropout(self.drop))
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if self.norm:
                layers.append(nn.BatchNorm1d(self.dims[i + 1]))
            layers.append(_ACTIVATIONS[self.activation]())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log-hazard scores.

        Parameters
        ----------
        x : torch.Tensor
            Latent representation, shape (batch, dims[0]).

        Returns
        -------
        torch.Tensor
            Log-hazard scores, shape (batch, 1).
        """
        return self.model(x)
