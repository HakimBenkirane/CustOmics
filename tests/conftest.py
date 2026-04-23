"""Shared fixtures for the customics test suite.

All fixtures use tiny synthetic data so tests run fast on CPU without any
external files.
"""

import numpy as np
import pandas as pd
import pytest
import torch


N_SAMPLES = 20
N_FEATURES_RNA = 50
N_FEATURES_CNV = 30
LATENT_DIM = 16
N_CLASSES = 3


@pytest.fixture(scope="session")
def sample_ids():
    return [f"SAMPLE_{i:03d}" for i in range(N_SAMPLES)]


@pytest.fixture(scope="session")
def rna_df(sample_ids):
    rng = np.random.default_rng(42)
    data = rng.random((N_SAMPLES, N_FEATURES_RNA)).astype(np.float32)
    return pd.DataFrame(data, index=sample_ids, columns=[f"gene_{i}" for i in range(N_FEATURES_RNA)])


@pytest.fixture(scope="session")
def cnv_df(sample_ids):
    rng = np.random.default_rng(43)
    data = rng.random((N_SAMPLES, N_FEATURES_CNV)).astype(np.float32)
    return pd.DataFrame(data, index=sample_ids, columns=[f"cnv_{i}" for i in range(N_FEATURES_CNV)])


@pytest.fixture(scope="session")
def omics_df(rna_df, cnv_df):
    return {"rna": rna_df, "cnv": cnv_df}


@pytest.fixture(scope="session")
def clinical_df(sample_ids):
    rng = np.random.default_rng(44)
    classes = ["TypeA", "TypeB", "TypeC"]
    labels = [classes[i % N_CLASSES] for i in range(N_SAMPLES)]
    return pd.DataFrame(
        {
            "label": labels,
            "OS": rng.integers(0, 2, size=N_SAMPLES),
            "OS.time": rng.integers(100, 2000, size=N_SAMPLES),
        },
        index=sample_ids,
    )


@pytest.fixture(scope="session")
def source_params():
    return {
        "rna": {
            "input_dim": N_FEATURES_RNA,
            "hidden_dim": [32],
            "latent_dim": LATENT_DIM,
            "norm": True,
            "dropout": 0.1,
        },
        "cnv": {
            "input_dim": N_FEATURES_CNV,
            "hidden_dim": [32],
            "latent_dim": LATENT_DIM,
            "norm": True,
            "dropout": 0.1,
        },
    }


@pytest.fixture(scope="session")
def central_params():
    return {
        "hidden_dim": [32],
        "latent_dim": LATENT_DIM,
        "norm": True,
        "dropout": 0.1,
        "beta": 1.0,
    }


@pytest.fixture(scope="session")
def classif_params():
    return {
        "n_class": N_CLASSES,
        "lambda": 1.0,
        "hidden_layers": [16],
        "dropout": 0.1,
    }


@pytest.fixture(scope="session")
def surv_params():
    return {
        "lambda": 1.0,
        "dims": [16],
        "activation": "SELU",
        "l2_reg": 1e-2,
        "norm": True,
        "dropout": 0.1,
    }


@pytest.fixture(scope="session")
def train_params():
    return {"switch": 2, "lr": 1e-3}


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def fitted_model(
    source_params, central_params, classif_params, surv_params, train_params,
    device, omics_df, clinical_df,
):
    from customics import CustOMICS

    model = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device,
    )
    model.fit(
        omics_train=omics_df,
        clinical_df=clinical_df,
        label="label",
        event="OS",
        surv_time="OS.time",
        n_epochs=3,
        batch_size=8,
        verbose=False,
    )
    return model
