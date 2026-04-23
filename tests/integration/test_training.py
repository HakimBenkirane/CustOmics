"""Integration test: full fit → evaluate pipeline on synthetic toy data."""

import numpy as np
import pytest
import torch

from customics import CustOMICS


def test_full_pipeline_classification(
    source_params, central_params, classif_params, surv_params, train_params,
    device, omics_df, clinical_df,
):
    """Fit a model and evaluate classification — smoke test for the full pipeline."""
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
        n_epochs=4,
        batch_size=8,
        verbose=False,
    )

    assert model._is_fitted
    assert len(model.history) == 4
    assert all(isinstance(h[0], float) for h in model.history)

    metrics = model.evaluate(
        omics_test=omics_df,
        clinical_df=clinical_df,
        label="label",
        event="OS",
        surv_time="OS.time",
        task="classification",
        batch_size=8,
    )
    assert isinstance(metrics, dict)
    assert "Accuracy" in metrics
    assert 0.0 <= metrics["Accuracy"] <= 1.0


def test_full_pipeline_survival(
    source_params, central_params, classif_params, surv_params, train_params,
    device, omics_df, clinical_df,
):
    """Fit a model and evaluate survival — smoke test for the full pipeline."""
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
    )
    ci = model.evaluate(
        omics_test=omics_df,
        clinical_df=clinical_df,
        label="label",
        event="OS",
        surv_time="OS.time",
        task="survival",
        batch_size=8,
    )
    assert isinstance(ci, float)
    assert 0.0 <= ci <= 1.0


def test_state_dict_save_load(
    source_params, central_params, classif_params, surv_params, train_params,
    device, omics_df, clinical_df, tmp_path,
):
    """Model weights must be fully recoverable via state_dict (verifies nn.ModuleList fix)."""
    model = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device,
    )
    model.fit(omics_df, clinical_df, label="label", event="OS", surv_time="OS.time",
              n_epochs=2, batch_size=8)

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    model2 = CustOMICS(
        source_params=source_params,
        central_params=central_params,
        classif_params=classif_params,
        surv_params=surv_params,
        train_params=train_params,
        device=device,
    )
    model2.load_state_dict(torch.load(path, map_location="cpu"))

    # Predictions must be identical after weight reload
    model2._is_fitted = True
    model2.label_encoder = model.label_encoder
    model2.one_hot_encoder = model.one_hot_encoder
    model2.baseline = model.baseline

    preds1 = model.predict(omics_df)
    preds2 = model2.predict(omics_df)
    np.testing.assert_array_equal(preds1, preds2)


def test_fit_with_validation(
    source_params, central_params, classif_params, surv_params, train_params,
    device, omics_df, clinical_df,
):
    """When omics_val is provided, history should contain (train, val) tuples."""
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
        omics_val=omics_df,
        n_epochs=2,
        batch_size=8,
    )
    assert all(len(h) == 2 for h in model.history)
