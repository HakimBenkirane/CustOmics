# Changelog

All notable changes to `customics` are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) — versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-04-21

First PyPI-ready release. The package was refactored from the original research
codebase with full packaging, typing, testing, and documentation.

### Added
- `pyproject.toml` — installable via `pip install customics`
- Proper `src/customics/` package layout with `__init__.py` in every sub-package
- `__version__ = "0.1.0"` version string
- `customics` CLI entry point (`python -m customics` or `customics` command)
- `customics.exceptions` module: `CustOmicsError`, `DataValidationError`,
  `ModelNotFittedError`, `ConfigurationError`
- Type hints on all public methods and class signatures
- `logging` integration throughout (replaces all `print()` calls)
- Input validation in `CustOMICS.__init__` and `fit()` with descriptive errors
- `CustOMICS.predict()` — dedicated classification prediction method
- `CustOMICS.source_names` attribute for explicit source-to-index mapping
- `CustOMICS._get_central_representation()` — always uses the central VAE for
  inference regardless of training phase
- `customics.visualization` module: `plot_loss`, `plot_representation`,
  `plot_survival_stratification` (extracted from the monolithic model class)
- `customics.explain.shap` module (replaces `ex_vae/shap_vae.py`)
- `MultiOmicsDataset.get_samples()` method
- Complete NumPy-style docstrings with `Parameters`, `Returns`, `Raises`, and
  `Examples` sections throughout
- Full test suite: unit tests for losses, metrics, dataset, encoders, model;
  integration tests for fit → evaluate → save/load
- GitHub Actions CI workflow (Python 3.9 and 3.11)
- `CHANGELOG.md` and `CONTRIBUTING.md`
- Updated `README.md` with installation, quick start, API table, and data-format
  documentation

### Fixed
- **[B1]** `self.autoencoders`, `self.lt_encoders`, `self.lt_decoders` were plain
  Python `list` objects — replaced with `nn.ModuleList` so `state_dict()`,
  `load_state_dict()`, and `to(device)` work correctly
- **[B2]** `eval('nn.{}()'.format(activation))` in `SurvivalNet` — replaced with
  a safe dictionary dispatch (`_ACTIVATIONS` map)
- **[B3]** `classification_loss` argument order was inverted (`y_true`/`y_pred`
  swapped relative to call sites) — corrected signature and docstring
- **[B4]** `FullyConnectedLayer` silently overwrote the passed `norm_layer`
  argument with `nn.BatchNorm1d` — fixed to honour the caller's choice
- **[B5]** `evaluate_latent()` called `svc_model.predict_proba()` without
  arguments — removed the broken method; `evaluate()` now correctly handles both
  tasks

### Changed
- Package name standardised to lowercase `customics` throughout
- All `from src.xxx import` statements updated to `from customics.xxx import`
- `Decoder` and `ProbabilisticDecoder` now reverse `hidden_dim` internally,
  mirroring the encoder ordering
- `fit()` now returns `self` for method chaining
- `CustOMICS.stratify()` and `CustOMICS.plot_*` are thin wrappers that delegate
  to `customics.visualization`
- `src/tools/prepare_dataset.py` and `src/tools/core_utils.py` removed (tightly
  coupled to internal TCGA paths; not library code)
- `src/debug/print_layer.py` removed (unused)
- `src/config/samples.txt` dependency removed from `utils.py`

### Deprecated
- `evaluate_latent()` — removed (was broken; use `evaluate()` instead)
