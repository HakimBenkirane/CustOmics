# Contributing to *customics*

Thank you for your interest in contributing.

You can contribute by:

- Opening an issue
- Discussing the current state of the code
- Making a Pull Request (PR)

## Quickstart

```sh
git clone https://github.com/HakimBenkirane/CustOmics.git
cd CustOmics
uv sync --dev
pre-commit install
uv run pytest
```

## Reporting Issues

Please open an issue at [github.com/HakimBenkirane/CustOmics/issues](https://github.com/HakimBenkirane/CustOmics/issues) and include:
- A minimal reproducible example
- Your Python and PyTorch versions
- The full traceback if applicable

## Making a Pull Request (PR)

To contribute code to **customics**:

1. Fork the repository
2. Install dependencies with `uv sync --dev`
3. Create a branch from `main`
4. Implement your changes according to the coding guidelines below
5. Open a pull request against `main` with a clear description

### Pull request checklist

- Explain what changed and why.
- Mention whether tests were added or updated.
- Confirm local tests passed.

## Coding Guidelines

### Styling and formatting

We use [`pre-commit`](https://pre-commit.com/) to run code quality checks before commits. This runs `ruff` and other minor checks.

Run checks manually if needed:

```sh
pre-commit run --all-files
```

General conventions:
- Follow the [PEP8](https://peps.python.org/pep-0008/) style guide.
- Use meaningful variable and function names.
- Add type hints to public function and class signatures.
- Add docstrings in NumPy style.
- Follow the existing style of the repository.

#### Project conventions

- **Imports** — use `from customics.xxx import yyy` (never `from src.xxx`).
- **Type hints** — add them to every public function and class signature.
- **Docstrings** — NumPy style with `Parameters`, `Returns`, and `Raises` sections.
- **Logging** — use `logger = logging.getLogger(__name__)` instead of `print()`.
- **No `eval()`** — use dict dispatch for dynamic object construction.
- **`nn.ModuleList`** — use it (not plain `list`) for collections of `nn.Module` objects.

## Testing

Tests run automatically on PRs. You should also run them locally before opening a PR.

Run all tests:

```sh
uv run pytest
```

Run coverage:

```sh
uv run pytest --cov --cov-config=pyproject.toml --cov-report=html --disable-warnings
open htmlcov/index.html
```

Run subsets:

```bash
uv run pytest tests/unit/
uv run pytest tests/integration/
```

All tests use synthetic data and run on CPU; no GPU or external files are required.

## Documentation

You can update documentation in `./docs`. Refer to [Zensical documentation](https://zensical.org/docs/get-started/) for details.

Serve docs locally:

```sh
uv run zensical serve
```
