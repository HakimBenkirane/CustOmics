---
name: customics package refactor complete
description: CustOmics was refactored from a research repo into a proper PyPI-ready package named customics (v0.1.0)
type: project
---

The repo was fully refactored from a research-only codebase into an installable `customics` package (v0.1.0).

**Why:** User wants to publish to PyPI.

**What was done:**
- New layout: `src/customics/` package with `__init__.py` in every sub-package
- `pyproject.toml` with `setuptools.build_meta` backend, Python ≥ 3.9
- Active conda env for this project: `bc_multiomics`
- Run tests with: `conda run -n bc_multiomics python -m pytest tests/`

**Key bugs fixed:**
- B1: `list` → `nn.ModuleList` for autoencoders (fixes `state_dict()`)
- B2: `eval()` → `_ACTIVATIONS` dict in `SurvivalNet`
- B3: `classification_loss` arg order corrected (`y_pred, y_true`)
- B4: `FullyConnectedLayer` now honours the passed `norm_layer`
- B5: Broken `evaluate_latent()` removed; `evaluate()` unified
- OHE fitted on integer-encoded labels (not strings) in `fit()`

**How to apply:** When working on this repo, install in editable mode in bc_multiomics and run pytest to verify.
