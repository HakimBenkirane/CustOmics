"""Standard autoencoder model."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from customics.encoders.encoder import Encoder
from customics.decoders.decoder import Decoder


class AutoEncoder(nn.Module):
    """Standard autoencoder for a single omics source.

    Parameters
    ----------
    encoder : Encoder
        Deterministic encoder network.
    decoder : Decoder
        Deterministic decoder network.
    device : torch.device
        Compute device.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x`` and reconstruct it.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstruction, shape (batch, input_dim).
        z : torch.Tensor
            Latent representation, shape (batch, latent_dim).
        """
        z = self.encoder(x)
        return self.decoder(z), z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to data space.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor, shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor, shape (batch, input_dim).
        """
        return self.decoder(z)

    def loss(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """Compute reconstruction loss (MSE).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        beta : float
            Unused here; kept for a consistent interface with :class:`VAE`.

        Returns
        -------
        torch.Tensor
            Scalar MSE reconstruction loss.
        """
        x_hat, _ = self.forward(x)
        return nn.MSELoss()(x, x_hat)
