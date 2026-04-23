# Contributing to customics

Thank you for your interest in contributing!

---

## Development Setup

```bash
git clone https://github.com/HakimBenkirane/CustOmics.git
cd CustOmics
pip install -e ".[dev]"
```

This installs the package in editable mode together with `pytest` and `pytest-cov`.

---

## Running Tests

```bash
pytest                          # run all tests
pytest tests/unit/              # unit tests only
pytest tests/integration/       # integration tests only
pytest --cov=customics          # with coverage report
```

All tests use synthetic data and run on CPU — no GPU or external files required.

---

## Code Conventions

- **Imports** — all internal imports use `from customics.xxx import yyy` (never `from src.xxx`).
- **Type hints** — add them to every public function and class signature.
- **Docstrings** — NumPy style with `Parameters`, `Returns`, and `Raises` sections.
  Add a short `Examples` block for public-facing methods.
- **Logging** — use `logger = logging.getLogger(__name__)` instead of `print()`.
- **No `eval()`** — use dict dispatch for dynamic object construction (see `_ACTIVATIONS` in `tasks/survival.py`).
- **`nn.ModuleList`** — use it (not plain `list`) for collections of `nn.Module` objects.

---

## Submitting Changes

1. Fork the repository and create a branch off `main`.
2. Make your changes, add tests for new behaviour, and ensure the full test suite passes.
3. Open a pull request against `main` with a clear description of what changed and why.

---

## Reporting Issues

Please open an issue at [github.com/HakimBenkirane/CustOmics/issues](https://github.com/HakimBenkirane/CustOmics/issues) and include:
- A minimal reproducible example
- Your Python and PyTorch versions
- The full traceback if applicable
