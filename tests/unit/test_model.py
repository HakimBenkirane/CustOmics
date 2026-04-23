"""Unit tests for the CustOMICS model class."""

import pytest
import torch

from customics import CustOMICS
from customics.exceptions import ConfigurationError, DataValidationError, ModelNotFittedError


class TestCustOMICSInstantiation:
    def test_builds_successfully(
        self, source_params, central_params, classif_params, surv_params, train_params, device
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        assert isinstance(model, CustOMICS)

    def test_parameter_count_positive(
        self, source_params, central_params, classif_params, surv_params, train_params, device
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        assert model.get_number_parameters() > 0

    def test_autoencoders_is_module_list(
        self, source_params, central_params, classif_params, surv_params, train_params, device
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        assert isinstance(model.autoencoders, torch.nn.ModuleList)

    def test_state_dict_contains_autoencoder_params(
        self, source_params, central_params, classif_params, surv_params, train_params, device
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        state = model.state_dict()
        # At least one key should belong to the autoencoders
        assert any("autoencoders" in k for k in state.keys())

    def test_invalid_dropout_raises(
        self, central_params, classif_params, surv_params, train_params, device
    ):
        bad_source = {
            "rna": {"input_dim": 50, "hidden_dim": [32], "latent_dim": 16,
                    "norm": True, "dropout": 1.5}
        }
        with pytest.raises(ConfigurationError, match="dropout"):
            CustOMICS(bad_source, central_params, classif_params, surv_params, train_params, device)

    def test_missing_n_class_raises(
        self, source_params, central_params, surv_params, train_params, device
    ):
        bad_classif = {"n_class": 1, "lambda": 1.0, "hidden_layers": [16], "dropout": 0.1}
        with pytest.raises(ConfigurationError, match="n_class"):
            CustOMICS(source_params, central_params, bad_classif, surv_params, train_params, device)

    def test_unfitted_predict_raises(
        self, source_params, central_params, classif_params, surv_params, train_params,
        device, omics_df,
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        with pytest.raises(ModelNotFittedError):
            model.predict(omics_df)


class TestCustOMICSFit:
    def test_fit_returns_self(
        self, source_params, central_params, classif_params, surv_params, train_params,
        device, omics_df, clinical_df,
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        result = model.fit(
            omics_df, clinical_df, label="label", event="OS", surv_time="OS.time",
            n_epochs=2, batch_size=8,
        )
        assert result is model

    def test_history_populated(
        self, source_params, central_params, classif_params, surv_params, train_params,
        device, omics_df, clinical_df,
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        model.fit(
            omics_df, clinical_df, label="label", event="OS", surv_time="OS.time",
            n_epochs=3, batch_size=8,
        )
        assert len(model.history) == 3

    def test_missing_label_column_raises(
        self, source_params, central_params, classif_params, surv_params, train_params,
        device, omics_df, clinical_df,
    ):
        model = CustOMICS(
            source_params, central_params, classif_params, surv_params, train_params, device
        )
        with pytest.raises(DataValidationError, match="not found"):
            model.fit(
                omics_df, clinical_df, label="nonexistent", event="OS", surv_time="OS.time",
            )


class TestCustOMICSInference:
    def test_get_latent_shape(self, fitted_model, omics_df):
        z = fitted_model.get_latent_representation(omics_df)
        assert z.shape[0] == 20  # N_SAMPLES

    def test_predict_shape(self, fitted_model, omics_df):
        preds = fitted_model.predict(omics_df)
        assert preds.shape == (20,)

    def test_predict_classes_valid(self, fitted_model, omics_df):
        preds = fitted_model.predict(omics_df)
        assert preds.min() >= 0
        assert preds.max() < 3  # N_CLASSES

    def test_evaluate_classification_returns_dict(
        self, fitted_model, omics_df, clinical_df
    ):
        result = fitted_model.evaluate(
            omics_df, clinical_df, label="label", event="OS", surv_time="OS.time",
            task="classification", batch_size=8,
        )
        assert isinstance(result, dict)
        assert "Accuracy" in result
        assert 0.0 <= result["Accuracy"] <= 1.0

    def test_evaluate_survival_returns_float(
        self, fitted_model, omics_df, clinical_df
    ):
        result = fitted_model.evaluate(
            omics_df, clinical_df, label="label", event="OS", surv_time="OS.time",
            task="survival", batch_size=8,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_invalid_task_raises(self, fitted_model, omics_df, clinical_df):
        with pytest.raises(ValueError, match="task must be"):
            fitted_model.evaluate(
                omics_df, clinical_df, label="label", event="OS", surv_time="OS.time",
                task="regression",
            )
