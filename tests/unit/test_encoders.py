"""Unit tests for encoder, decoder, and model building blocks."""

import torch
import pytest

from customics.encoders.encoder import Encoder
from customics.encoders.probabilistic_encoder import ProbabilisticEncoder
from customics.decoders.decoder import Decoder
from customics.decoders.probabilistic_decoder import ProbabilisticDecoder
from customics.models.autoencoder import AutoEncoder
from customics.models.vae import VAE
from customics.tasks.survival import SurvivalNet
from customics.tasks.classification import MultiClassifier
from customics.exceptions import ConfigurationError


BATCH = 8
INPUT_DIM = 64
HIDDEN_DIM = [32]
LATENT_DIM = 16
DEVICE = torch.device("cpu")


class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        out = enc(torch.randn(BATCH, INPUT_DIM))
        assert out.shape == (BATCH, LATENT_DIM)

    def test_gradients_flow(self):
        enc = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        x = torch.randn(BATCH, INPUT_DIM, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestProbabilisticEncoder:
    def test_output_shapes(self):
        enc = ProbabilisticEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        mean, log_var = enc(torch.randn(BATCH, INPUT_DIM))
        assert mean.shape == (BATCH, LATENT_DIM)
        assert log_var.shape == (BATCH, LATENT_DIM)


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
        out = dec(torch.randn(BATCH, LATENT_DIM))
        assert out.shape == (BATCH, INPUT_DIM)


class TestProbabilisticDecoder:
    def test_output_range(self):
        dec = ProbabilisticDecoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
        out = dec(torch.randn(BATCH, LATENT_DIM))
        assert out.shape == (BATCH, INPUT_DIM)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


class TestAutoEncoder:
    def test_forward_shapes(self):
        enc = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        dec = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
        ae = AutoEncoder(enc, dec, DEVICE)
        x_hat, z = ae(torch.randn(BATCH, INPUT_DIM))
        assert x_hat.shape == (BATCH, INPUT_DIM)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_loss_is_positive(self):
        enc = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        dec = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
        ae = AutoEncoder(enc, dec, DEVICE)
        x = torch.randn(BATCH, INPUT_DIM)
        loss = ae.loss(x, beta=1.0)
        assert loss.item() > 0


class TestVAE:
    def test_forward_shapes(self):
        enc = ProbabilisticEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        dec = ProbabilisticDecoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
        vae = VAE(enc, dec, DEVICE)
        x_hat, z = vae(torch.randn(BATCH, INPUT_DIM))
        assert x_hat.shape == (BATCH, INPUT_DIM)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_loss_is_positive(self):
        enc = ProbabilisticEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        dec = ProbabilisticDecoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
        vae = VAE(enc, dec, DEVICE)
        loss = vae.loss(torch.randn(BATCH, INPUT_DIM), beta=1.0)
        assert loss.item() > 0


class TestSurvivalNet:
    def test_output_shape(self):
        net = SurvivalNet({"dims": [16, 8, 1], "drop": 0.1, "norm": False, "activation": "SELU"})
        out = net(torch.randn(BATCH, 16))
        assert out.shape == (BATCH, 1)

    def test_invalid_activation_raises(self):
        with pytest.raises(ConfigurationError, match="Unsupported activation"):
            SurvivalNet({"dims": [16, 1], "drop": 0.0, "norm": False, "activation": "invalid"})


class TestMultiClassifier:
    def test_output_shape(self):
        clf = MultiClassifier(n_class=4, latent_dim=LATENT_DIM, class_dim=[16])
        out = clf(torch.randn(BATCH, LATENT_DIM))
        assert out.shape == (BATCH, 4)

    def test_predict_shape(self):
        clf = MultiClassifier(n_class=3, latent_dim=LATENT_DIM, class_dim=[16])
        pred = clf.predict(torch.randn(BATCH, LATENT_DIM))
        assert pred.shape == (BATCH,)
        assert pred.max().item() < 3
