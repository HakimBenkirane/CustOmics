"""Building blocks for fully-connected neural network layers."""

from __future__ import annotations

import torch.nn as nn


_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "no": None,  # type: ignore[assignment]
}


class FullyConnectedLayer(nn.Module):
    """A single fully-connected block: Linear → [BatchNorm] → [Dropout] → [Activation].

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output feature dimension.
    norm_layer : type or bool, optional
        Normalization layer class (e.g. ``nn.BatchNorm1d``) or ``True`` to use
        the default ``nn.BatchNorm1d``.  Pass ``False`` or ``None`` to skip
        normalisation (equivalent to ``normalization=False``).
    leaky_slope : float
        Negative slope for LeakyReLU.
    dropout : float
        Dropout probability; disabled when ``<= 0``.
    activation : bool
        Whether to append an activation function.
    normalization : bool
        Whether to append a batch-normalisation layer.
    activation_name : str
        Name of the activation function (case-insensitive).  Supported values:
        ``'relu'``, ``'sigmoid'``, ``'leakyrelu'``, ``'tanh'``, ``'softmax'``,
        ``'no'``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        norm_layer: type[nn.Module] | bool = nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0.2,
        activation: bool = True,
        normalization: bool = True,
        activation_name: str = "LeakyReLU",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]

        if normalization:
            if isinstance(norm_layer, type):
                layers.append(norm_layer(output_dim))
            else:
                layers.append(nn.BatchNorm1d(output_dim))

        if 0 < dropout <= 1:
            layers.append(nn.Dropout(p=dropout))

        if activation:
            key = activation_name.lower()
            if key not in _ACTIVATION_MAP:
                raise ValueError(
                    f"Activation '{activation_name}' is not supported. "
                    f"Choose from {list(_ACTIVATION_MAP)}."
                )
            act_cls = _ACTIVATION_MAP[key]
            if act_cls is not None:
                layers.append(
                    act_cls(negative_slope=leaky_slope, inplace=True)
                    if key == "leakyrelu"
                    else (act_cls(dim=1) if key == "softmax" else act_cls())
                )

        self.fc_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_block(x)
