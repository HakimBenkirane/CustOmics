"""Core customics model: hierarchical multi-omics integration with multi-task learning."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader

from customics.datasets.multi_omics_dataset import MultiOmicsDataset
from customics.decoders.decoder import Decoder
from customics.decoders.probabilistic_decoder import ProbabilisticDecoder
from customics.encoders.encoder import Encoder
from customics.encoders.probabilistic_encoder import ProbabilisticEncoder
from customics.exceptions import ConfigurationError, DataValidationError, ModelNotFittedError
from customics.loss.classification_loss import classification_loss
from customics.loss.survival_loss import CoxLoss
from customics.metrics.classification import multi_classification_evaluation, plot_roc_multiclass
from customics.metrics.survival import CIndex_lifeline
from customics.models.autoencoder import AutoEncoder
from customics.models.vae import VAE
from customics.tasks.classification import MultiClassifier
from customics.tasks.survival import SurvivalNet
from customics.tools.utils import get_common_samples

logger = logging.getLogger(__name__)


class CustOMICS(nn.Module):
    """Multi-omics integration model with hierarchical autoencoders and multi-task learning.

    The architecture has two stages:

    1. **Per-source autoencoders** — one deterministic autoencoder per omics
       modality compresses high-dimensional features into a latent vector.
    2. **Central VAE** — a variational autoencoder integrates the concatenated
       per-source representations into a shared latent space used by downstream
       task heads (classifier and survival predictor).

    Training uses two phases, controlled by ``train_params['switch']``:

    - **Phase 1** (``epoch < switch``): only per-source autoencoders and task
      heads are optimised; the central VAE is kept fixed.
    - **Phase 2** (``epoch >= switch``): the full pipeline — per-source AEs,
      central VAE, and task heads — is jointly optimised.

    Parameters
    ----------
    source_params : dict
        Per-source configuration.  Keys are source names; each value is a dict
        with the following keys:

        * ``input_dim`` (int): number of input features.
        * ``hidden_dim`` (list of int): hidden layer sizes.
        * ``latent_dim`` (int): per-source latent dimension.
        * ``norm`` (bool): whether to use batch normalisation.
        * ``dropout`` (float): dropout rate in ``[0, 1]``.

    central_params : dict
        Central VAE configuration.  Required keys: ``hidden_dim`` (list of
        int), ``latent_dim`` (int), ``norm`` (bool), ``dropout`` (float),
        ``beta`` (float — MMD regularisation weight).

    classif_params : dict
        Classifier configuration.  Required keys: ``n_class`` (int, >= 2),
        ``lambda`` (float — loss weight), ``hidden_layers`` (list of int),
        ``dropout`` (float).

    surv_params : dict
        Survival-predictor configuration.  Required keys: ``lambda`` (float),
        ``dims`` (list of int), ``activation`` (str), ``l2_reg`` (float),
        ``norm`` (bool), ``dropout`` (float).

    train_params : dict
        Training hyperparameters.  Required keys: ``switch`` (int — epoch at
        which to enter phase 2) and ``lr`` (float — learning rate).

    device : torch.device
        Compute device.

    Examples
    --------
    >>> import torch
    >>> from customics import CustOMICS
    >>> model = CustOMICS(
    ...     source_params={
    ...         "rna": {"input_dim": 1000, "hidden_dim": [256, 128],
    ...                 "latent_dim": 64, "norm": True, "dropout": 0.2},
    ...     },
    ...     central_params={"hidden_dim": [128], "latent_dim": 32,
    ...                     "norm": True, "dropout": 0.2, "beta": 1},
    ...     classif_params={"n_class": 3, "lambda": 5,
    ...                     "hidden_layers": [32, 16], "dropout": 0.2},
    ...     surv_params={"lambda": 1, "dims": [32, 16], "activation": "SELU",
    ...                  "l2_reg": 1e-2, "norm": True, "dropout": 0.2},
    ...     train_params={"switch": 5, "lr": 1e-3},
    ...     device=torch.device("cpu"),
    ... )
    """

    def __init__(
        self,
        source_params: Dict[str, Dict],
        central_params: Dict,
        classif_params: Dict,
        surv_params: Dict,
        train_params: Dict,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._validate_params(source_params, central_params, classif_params, train_params)

        self.device = device
        self.source_names: List[str] = list(source_params.keys())
        self.n_source = len(self.source_names)
        self.beta = central_params["beta"]
        self.num_classes = classif_params["n_class"]
        self.lambda_classif = classif_params["lambda"]
        self.lambda_survival = surv_params["lambda"]
        self.switch_epoch = train_params["switch"]
        self.lr = train_params["lr"]
        self.phase = 1
        self._is_fitted = False

        # Store config dicts so save() can serialise them.
        self._source_params = source_params
        self._central_params = central_params
        self._classif_params = classif_params
        self._surv_params = surv_params
        self._train_params = train_params

        # ------------------------------------------------------------------ #
        # Per-source autoencoders (nn.ModuleList so PyTorch tracks them)
        # ------------------------------------------------------------------ #
        self.autoencoders = nn.ModuleList(
            [
                AutoEncoder(
                    encoder=Encoder(
                        input_dim=source_params[s]["input_dim"],
                        hidden_dim=source_params[s]["hidden_dim"],
                        latent_dim=source_params[s]["latent_dim"],
                        norm_layer=source_params[s]["norm"],
                        dropout=source_params[s]["dropout"],
                    ),
                    decoder=Decoder(
                        latent_dim=source_params[s]["latent_dim"],
                        hidden_dim=source_params[s]["hidden_dim"],
                        output_dim=source_params[s]["input_dim"],
                        norm_layer=source_params[s]["norm"],
                        dropout=source_params[s]["dropout"],
                    ),
                    device=device,
                )
                for s in self.source_names
            ]
        )

        # ------------------------------------------------------------------ #
        # Central VAE
        # ------------------------------------------------------------------ #
        self.rep_dim = sum(source_params[s]["latent_dim"] for s in self.source_names)
        self.central_layer = VAE(
            encoder=ProbabilisticEncoder(
                input_dim=self.rep_dim,
                hidden_dim=central_params["hidden_dim"],
                latent_dim=central_params["latent_dim"],
                norm_layer=central_params["norm"],
                dropout=central_params["dropout"],
            ),
            decoder=ProbabilisticDecoder(
                latent_dim=central_params["latent_dim"],
                hidden_dim=central_params["hidden_dim"],
                output_dim=self.rep_dim,
                norm_layer=central_params["norm"],
                dropout=central_params["dropout"],
            ),
            device=device,
        )

        # ------------------------------------------------------------------ #
        # Task heads
        # ------------------------------------------------------------------ #
        self.classifier = MultiClassifier(
            n_class=self.num_classes,
            latent_dim=central_params["latent_dim"],
            dropout=classif_params["dropout"],
            class_dim=classif_params["hidden_layers"],
        )
        self.survival_predictor = SurvivalNet(
            {
                "drop": surv_params["dropout"],
                "norm": surv_params["norm"],
                "dims": [central_params["latent_dim"]] + surv_params["dims"] + [1],
                "activation": surv_params["activation"],
            }
        )

        self._relocate()
        self.optimizer = self._build_optimizer()

        # Filled during fit()
        self.history: List[Tuple] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.baseline = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_params(
        source_params: Dict,
        central_params: Dict,
        classif_params: Dict,
        train_params: Dict,
    ) -> None:
        if not source_params:
            raise ConfigurationError("source_params must contain at least one source.")
        for name, sp in source_params.items():
            for key in ("input_dim", "hidden_dim", "latent_dim", "norm", "dropout"):
                if key not in sp:
                    raise ConfigurationError(
                        f"source_params['{name}'] is missing required key '{key}'."
                    )
            if not 0.0 <= sp["dropout"] <= 1.0:
                raise ConfigurationError(
                    f"dropout for source '{name}' must be in [0, 1], got {sp['dropout']}."
                )
        if classif_params.get("n_class", 0) < 2:
            raise ConfigurationError("classif_params['n_class'] must be >= 2.")
        for key in ("switch", "lr"):
            if key not in train_params:
                raise ConfigurationError(f"train_params is missing required key '{key}'.")

    def _build_optimizer(self) -> Adam:
        return Adam(self.parameters(), lr=self.lr)

    def _relocate(self) -> None:
        self.autoencoders.to(self.device)
        self.central_layer.to(self.device)
        self.classifier.to(self.device)
        self.survival_predictor.to(self.device)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError(
                "This CustOMICS instance has not been fitted yet. Call fit() first."
            )

    # ------------------------------------------------------------------ #
    # Phase management
    # ------------------------------------------------------------------ #

    def _switch_phase(self, epoch: int) -> None:
        self.phase = 1 if epoch < self.switch_epoch else 2

    # ------------------------------------------------------------------ #
    # Forward pass (nn.Module interface)
    # ------------------------------------------------------------------ #

    def forward(
        self, x: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Full forward pass through per-source AEs and central encoder.

        Parameters
        ----------
        x : list of torch.Tensor
            One tensor per omics source, shape ``(batch, features_i)``.

        Returns
        -------
        lt_hat : list of torch.Tensor
            Per-source reconstructions.
        lt_rep : list of torch.Tensor
            Per-source latent vectors.
        mean : torch.Tensor
            Central VAE posterior mean, shape ``(batch, central_latent_dim)``.
        """
        lt_hat, lt_rep = [], []
        for xi, ae in zip(x, self.autoencoders):
            hat, rep = ae(xi)
            lt_hat.append(hat)
            lt_rep.append(rep)
        mean, _ = self.central_layer.encoder(torch.cat(lt_rep, dim=1))
        return lt_hat, lt_rep, mean

    # ------------------------------------------------------------------ #
    # Internal helpers for training
    # ------------------------------------------------------------------ #

    def _compute_training_loss(
        self, x: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(latent_z, reconstruction_loss)`` for the current phase.

        Phase 1 returns the first source's representation; phase 2 returns the
        central VAE mean.  Both are used as input to the task heads during
        training.
        """
        lt_rep: List[torch.Tensor] = []
        recon_loss = torch.tensor(0.0, device=self.device)
        for xi, ae in zip(x, self.autoencoders):
            _, rep = ae(xi)
            lt_rep.append(rep)
            recon_loss = recon_loss + ae.loss(xi, self.beta)

        if self.phase == 1:
            return lt_rep, recon_loss

        central_concat = torch.cat(lt_rep, dim=1)
        recon_loss = recon_loss + self.central_layer.loss(central_concat, self.beta)
        mean, _ = self.central_layer.encoder(central_concat)
        return mean, recon_loss

    def _get_central_representation(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Always return the central VAE mean (used for inference)."""
        lt_rep = [ae(xi)[1] for xi, ae in zip(x, self.autoencoders)]
        mean, _ = self.central_layer.encoder(torch.cat(lt_rep, dim=1))
        return mean

    def _set_train_mode(self) -> None:
        self.autoencoders.train()
        self.central_layer.train()
        self.classifier.train()
        self.survival_predictor.train()

    def _set_eval_mode(self) -> None:
        self.autoencoders.eval()
        self.central_layer.eval()
        self.classifier.eval()
        self.survival_predictor.eval()

    def _train_step(
        self,
        x: List[torch.Tensor],
        labels: torch.Tensor,
        os_time: torch.Tensor,
        os_event: torch.Tensor,
    ) -> torch.Tensor:
        x = [xi.to(self.device) for xi in x]
        self.optimizer.zero_grad()

        z_or_reps, recon_loss = self._compute_training_loss(x)

        if self.phase == 1:
            # Apply task heads to every per-source representation separately
            task_loss = torch.tensor(0.0, device=self.device)
            for z in z_or_reps:
                task_loss = task_loss + self.lambda_survival * CoxLoss(
                    os_time, os_event, self.survival_predictor(z), self.device
                )
                task_loss = task_loss + self.lambda_classif * classification_loss(
                    "CE", self.classifier(z), labels
                )
        else:
            z = z_or_reps
            task_loss = (
                self.lambda_survival * CoxLoss(os_time, os_event, self.survival_predictor(z), self.device)
                + self.lambda_classif * classification_loss("CE", self.classifier(z), labels)
            )

        return recon_loss + task_loss

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        total, n = 0.0, 0
        for x, labels, os_time, os_event in loader:
            if training:
                self._set_train_mode()
                loss = self._train_step(x, labels, os_time, os_event)
                loss.backward()
                self.optimizer.step()
            else:
                self._set_eval_mode()
                with torch.no_grad():
                    loss = self._train_step(x, labels, os_time, os_event)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    # Public training API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        omics_train: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        label: str,
        event: str,
        surv_time: str,
        omics_val: Optional[Dict[str, pd.DataFrame]] = None,
        batch_size: int = 32,
        n_epochs: int = 30,
        verbose: bool = False,
    ) -> "CustOMICS":
        """Train the customics model.

        Parameters
        ----------
        omics_train : dict of str → pd.DataFrame
            Training omics data.  Each DataFrame must be indexed by sample ID.
        clinical_df : pd.DataFrame
            Clinical metadata indexed by sample ID.
        label : str
            Column in ``clinical_df`` containing class labels.
        event : str
            Column in ``clinical_df`` containing the event indicator (0/1).
        surv_time : str
            Column in ``clinical_df`` containing survival time.
        omics_val : dict, optional
            Validation omics data; same format as ``omics_train``.
        batch_size : int
            Mini-batch size.
        n_epochs : int
            Number of training epochs.
        verbose : bool
            Log epoch-level loss when True.

        Returns
        -------
        CustOMICS
            ``self`` (enables method chaining).

        Raises
        ------
        DataValidationError
            If required columns are missing or samples don't overlap.
        """
        self._validate_fit_inputs(omics_train, clinical_df, label, event, surv_time)

        encoded_clinical = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(encoded_clinical[label].values)
        encoded_clinical[label] = self.label_encoder.transform(
            encoded_clinical[label].values
        )
        # Fit OHE on integer-encoded labels so it can transform integer y_true at eval time
        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(
            encoded_clinical[label].values.reshape(-1, 1)
        )

        loader_kw: Dict = (
            {"num_workers": 2, "pin_memory": True} if self.device.type == "cuda" else {}
        )

        lt_train = get_common_samples(list(omics_train.values()) + [clinical_df])
        self.baseline = self._compute_baseline(clinical_df, lt_train, event, surv_time)
        train_loader = DataLoader(
            MultiOmicsDataset(omics_train, encoded_clinical, lt_train, label, event, surv_time),
            batch_size=batch_size,
            shuffle=True,
            **loader_kw,
        )

        val_loader: Optional[DataLoader] = None
        if omics_val is not None:
            lt_val = get_common_samples(list(omics_val.values()) + [clinical_df])
            val_loader = DataLoader(
                MultiOmicsDataset(omics_val, encoded_clinical, lt_val, label, event, surv_time),
                batch_size=batch_size,
                shuffle=False,
                **loader_kw,
            )

        self.history = []
        for epoch in range(n_epochs):
            self._switch_phase(epoch)
            train_loss = self._run_epoch(train_loader, training=True)
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
                self.history.append((train_loss, val_loss))
                if verbose:
                    logger.info(
                        "Epoch %d/%d | train=%.4f | val=%.4f",
                        epoch + 1, n_epochs, train_loss, val_loss,
                    )
            else:
                self.history.append((train_loss,))
                if verbose:
                    logger.info(
                        "Epoch %d/%d | train=%.4f", epoch + 1, n_epochs, train_loss
                    )

        self._is_fitted = True
        return self

    def _validate_fit_inputs(
        self,
        omics_train: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        label: str,
        event: str,
        surv_time: str,
    ) -> None:
        for col in (label, event, surv_time):
            if col not in clinical_df.columns:
                raise DataValidationError(
                    f"Column '{col}' not found in clinical_df. "
                    f"Available: {list(clinical_df.columns)}."
                )
        for source, df in omics_train.items():
            overlap = set(df.index) & set(clinical_df.index)
            if not overlap:
                raise DataValidationError(
                    f"Source '{source}' shares no sample IDs with clinical_df."
                )

    def _compute_baseline(
        self, clinical_df: pd.DataFrame, lt_samples: List[str], event: str, surv_time: str
    ):
        kmf = KaplanMeierFitter()
        kmf.fit(clinical_df.loc[lt_samples, surv_time], clinical_df.loc[lt_samples, event])
        return kmf.survival_function_

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def get_latent_representation(
        self,
        omics_df: Dict[str, pd.DataFrame],
    ) -> np.ndarray:
        """Compute the integrated central latent representation.

        Parameters
        ----------
        omics_df : dict of str → pd.DataFrame
            Omics data for all sources (same keys as used in ``fit``).

        Returns
        -------
        np.ndarray
            Latent matrix, shape ``(n_samples, central_latent_dim)``.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        self._require_fitted()
        self._set_eval_mode()
        x = [
            torch.tensor(omics_df[s].values, dtype=torch.float32).to(self.device)
            for s in self.source_names
        ]
        with torch.no_grad():
            z = self._get_central_representation(x)
        return z.cpu().numpy()

    def predict(self, omics_df: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        omics_df : dict
            Omics data matching the sources used in ``fit``.

        Returns
        -------
        np.ndarray
            Integer class predictions, shape ``(n_samples,)``.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        self._require_fitted()
        self._set_eval_mode()
        z = torch.tensor(self.get_latent_representation(omics_df), dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            logits = self.classifier(z)
        return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_survival(
        self, omics_df: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Compute patient-level estimated survival functions.

        Parameters
        ----------
        omics_df : dict
            Omics data matching the sources used in ``fit``.

        Returns
        -------
        dict
            Maps sample ID → estimated survival function DataFrame.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        self._require_fitted()
        lt_samples = get_common_samples(list(omics_df.values()))
        z = torch.tensor(
            self.get_latent_representation(omics_df), dtype=torch.float32
        ).to(self.device)
        self._set_eval_mode()
        with torch.no_grad():
            risk_scores = self.survival_predictor(z).cpu().numpy()
        return {s: self.baseline * np.exp(r[0]) for s, r in zip(lt_samples, risk_scores)}

    def source_predict(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """Predict class logits from a single-source input tensor.

        Used internally by the SHAP :class:`~customics.explain.shap.ModelWrapper`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for ``source``, shape ``(batch, features)``.
        source : str
            Source name (must be in ``self.source_names``).

        Returns
        -------
        torch.Tensor
            Class logits, shape ``(batch, n_class)``.
        """
        if source not in self.source_names:
            raise ValueError(
                f"Source '{source}' not recognised. Known sources: {self.source_names}."
            )
        idx = self.source_names.index(source)
        z = self.autoencoders[idx].encoder(x)
        return self.classifier(z)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        omics_test: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        label: str,
        event: str,
        surv_time: str,
        task: str,
        batch_size: int = 32,
        plot_roc: bool = False,
    ) -> Union[float, Dict[str, float]]:
        """Evaluate the model on held-out data.

        Parameters
        ----------
        omics_test : dict
            Test omics data.
        clinical_df : pd.DataFrame
            Clinical metadata.
        label : str
            Class-label column.
        event : str
            Event-indicator column.
        surv_time : str
            Survival-time column.
        task : str
            ``'classification'`` or ``'survival'``.
        batch_size : int
            Evaluation batch size.
        plot_roc : bool
            Save a ROC curve image (classification only).

        Returns
        -------
        float
            Concordance index for ``task='survival'``.
        dict
            Metrics dict (Accuracy, F1-score, Precision, Recall, AUC) for
            ``task='classification'``.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        ValueError
            If ``task`` is not ``'classification'`` or ``'survival'``.
        """
        self._require_fitted()
        if task not in ("classification", "survival"):
            raise ValueError(
                f"task must be 'classification' or 'survival', got '{task}'."
            )

        encoded_clinical = clinical_df.copy()
        encoded_clinical[label] = self.label_encoder.transform(
            encoded_clinical[label].values
        )

        loader_kw: Dict = (
            {"num_workers": 2, "pin_memory": True} if self.device.type == "cuda" else {}
        )
        lt_samples = get_common_samples(list(omics_test.values()) + [clinical_df])
        test_loader = DataLoader(
            MultiOmicsDataset(omics_test, encoded_clinical, lt_samples, label, event, surv_time),
            batch_size=batch_size,
            shuffle=False,
            **loader_kw,
        )

        self._set_eval_mode()
        all_y_true, all_y_pred, all_y_proba = [], [], []
        all_hazard, all_os_time, all_os_event = [], [], []

        with torch.no_grad():
            for x, labels, os_time, os_event in test_loader:
                x = [xi.to(self.device) for xi in x]
                z = self._get_central_representation(x)

                if task == "survival":
                    hazard = self.survival_predictor(z).cpu().numpy().reshape(-1, 1)
                    all_hazard.append(hazard)
                    all_os_time.append(os_time.numpy())
                    all_os_event.append(os_event.numpy())
                else:
                    logits = self.classifier(z)
                    all_y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
                    all_y_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
                    all_y_true.append(labels.cpu().numpy())

        if task == "survival":
            hazard_cat = np.vstack(all_hazard)
            return float(CIndex_lifeline(
                hazard_cat,
                np.concatenate(all_os_event),
                np.concatenate(all_os_time),
            ))

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_proba = np.vstack(all_y_proba)
        if plot_roc:
            plot_roc_multiclass(
                y_test=y_true,
                y_pred_proba=y_proba,
                filename="test",
                n_classes=self.num_classes,
                var_names=np.unique(clinical_df[label].values.tolist()).tolist(),
            )
        return multi_classification_evaluation(
            y_true, y_pred, y_proba, ohe=self.one_hot_encoder
        )

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample_id: List[str],
        omics_df: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        source: str,
        subtype: str,
        label: str = "PAM50",
        device: str = "cpu",
        show: bool = False,
    ) -> None:
        """Compute and plot SHAP values for one omics source and one subtype.

        Uses :class:`shap.DeepExplainer` on the single-source forward path.
        SHAP values are computed for ``subtype`` against all other classes.

        Parameters
        ----------
        sample_id : list of str
            Sample IDs to use as the SHAP background and foreground sets.
        omics_df : dict
            Multi-omics data.
        clinical_df : pd.DataFrame
            Clinical metadata.
        source : str
            Omics source key to explain.
        subtype : str
            Class label to explain.
        label : str
            Column in ``clinical_df`` with class labels.
        device : str
            Device for SHAP tensors.
        show : bool
            Display the SHAP plot interactively.

        Raises
        ------
        ModelNotFittedError
            If called before ``fit()``.
        """
        import shap
        import matplotlib.pyplot as plt
        from customics.explain.shap import (
            ModelWrapper,
            addToTensor,
            processPhenotypeDataForSamples,
            randomTrainingSample,
            splitExprandSample,
        )

        self._require_fitted()
        expr_df = omics_df[source]
        sample_id = list(set(sample_id) & set(expr_df.index))
        phenotype = processPhenotypeDataForSamples(clinical_df, sample_id, self.label_encoder)
        condition = phenotype[label] == subtype

        expr_df = expr_df.loc[sample_id, :]
        background = addToTensor(randomTrainingSample(expr_df, 10), device)
        foreground = addToTensor(
            splitExprandSample(condition=condition, sample_size=10, expr=expr_df), device
        )

        explainer = shap.DeepExplainer(ModelWrapper(self, source=source), background)
        shap_values = explainer.shap_values(foreground, ranked_outputs=None)

        tumour_expr = expr_df.head(10)
        shap.summary_plot(
            shap_values[0],
            features=tumour_expr,
            feature_names=list(expr_df.columns),
            show=False,
            plot_type="violin",
            max_display=10,
            plot_size=[4, 6],
        )
        plt.savefig(f"shap_{source}_{subtype}.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def get_number_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def plot_loss(self, show: bool = True) -> None:
        """Plot training (and validation) loss history.

        Parameters
        ----------
        show : bool
            Display the figure interactively.
        """
        from customics.visualization import plot_loss as _plot

        _plot(self.history, self.switch_epoch, show=show)

    def plot_representation(
        self,
        omics_df: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        label: str,
        filename: str,
        title: str,
        show: bool = True,
    ) -> None:
        """Compute latent representations and save a t-SNE scatter plot.

        Parameters
        ----------
        omics_df : dict
            Multi-omics data.
        clinical_df : pd.DataFrame
            Clinical metadata.
        label : str
            Column to use for colouring samples.
        filename : str
            Output path prefix (a ``.png`` suffix is appended).
        title : str
            Plot title.
        show : bool
            Display the figure interactively.
        """
        from customics.visualization import plot_representation as _plot

        _plot(self, omics_df, clinical_df, label, filename, title, show=show)

    def stratify(
        self,
        omics_df: Dict[str, pd.DataFrame],
        clinical_df: pd.DataFrame,
        event: str,
        surv_time: str,
        plot_title: str = "",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Stratify patients by predicted risk and plot Kaplan-Meier curves.

        Parameters
        ----------
        omics_df : dict
            Multi-omics data.
        clinical_df : pd.DataFrame
            Clinical metadata.
        event : str
            Event-indicator column.
        surv_time : str
            Survival-time column.
        plot_title : str
            Title prefix for the figure.
        save_path : str, optional
            Save the figure to this path when provided.
        show : bool
            Display the figure interactively.
        """
        from customics.visualization import plot_survival_stratification as _plot

        _plot(self, omics_df, clinical_df, event, surv_time, plot_title, save_path, show)

    # ---------------------------------------------------------------------- #
    # Serialisation
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save the model architecture config and trained weights to *path*.

        The checkpoint contains the five parameter dicts needed to reconstruct
        the model plus the ``state_dict`` and fitted encoders so that inference
        works immediately after :meth:`load`.

        Parameters
        ----------
        path : str
            Destination file (conventionally ``*.pt`` or ``*.pth``).
        """
        checkpoint = {
            "source_params":   self._source_params,
            "central_params":  self._central_params,
            "classif_params":  self._classif_params,
            "surv_params":     self._surv_params,
            "train_params":    self._train_params,
            "state_dict":      self.state_dict(),
            "label_encoder":   self.label_encoder,
            "one_hot_encoder": self.one_hot_encoder,
            "history":         self.history,
            "_is_fitted":      self._is_fitted,
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "CustOMICS":
        """Load a model previously saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the checkpoint file written by :meth:`save`.
        device : torch.device, optional
            Target device.  Defaults to CPU when not specified.

        Returns
        -------
        CustOMICS
            A fully initialised, ready-to-use model instance.
        """
        if device is None:
            device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            source_params  = checkpoint["source_params"],
            central_params = checkpoint["central_params"],
            classif_params = checkpoint["classif_params"],
            surv_params    = checkpoint["surv_params"],
            train_params   = checkpoint["train_params"],
            device         = device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.label_encoder   = checkpoint.get("label_encoder")
        model.one_hot_encoder = checkpoint.get("one_hot_encoder")
        model.history         = checkpoint.get("history", [])
        model._is_fitted      = checkpoint.get("_is_fitted", True)
        model.to(device)
        logger.info("Model loaded from %s", path)
        return model
