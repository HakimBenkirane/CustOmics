"""Variational autoencoder model with MMD regularisation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from customics.encoders.probabilistic_encoder import ProbabilisticEncoder
from customics.decoders.probabilistic_decoder import ProbabilisticDecoder
from customics.loss.mmd_loss import compute_mmd


class VAE(nn.Module):
    """Variational autoencoder using Maximum Mean Discrepancy (MMD) regularisation.

    The VAE loss combines MSE reconstruction with an MMD penalty between the
    learned posterior and a standard Gaussian prior.

    Parameters
    ----------
    encoder : ProbabilisticEncoder
        Inference network that outputs ``(mean, log_var)``.
    decoder : ProbabilisticDecoder
        Generative network.
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        encoder: ProbabilisticEncoder,
        decoder: ProbabilisticDecoder,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.to(device)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterisation trick.

        Parameters
        ----------
        mean : torch.Tensor
            Posterior mean.
        log_var : torch.Tensor
            Posterior log-variance.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        return mean + std * eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x``, sample ``z``, and reconstruct.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstruction, shape (batch, input_dim).
        z : torch.Tensor
            Sampled latent vector, shape (batch, latent_dim).
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), z

    def loss(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """Compute the VAE loss: reconstruction + ``beta`` × MMD.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        beta : float
            Weight for the MMD regularisation term.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        x_hat, z = self.forward(x)
        recon = nn.MSELoss()(x, x_hat)
        prior_samples = torch.randn_like(z).to(self.device)
        mmd = compute_mmd(prior_samples, z)
        return recon + beta * mmd
