# Contributing

Use Python 3.10+ and install `python -m pip install -e ".[dev]"` in a virtual environment. The `src/` layout ensures tests exercise an installed package.

Before submitting a change:

```bash
ruff check .
ruff format --check .
python -m pytest -m "not torch and not tf"
python -m build
python -m twine check dist/*
```

Test transport changes with real localhost sockets. Keep ML imports lazy so the core package remains usable without either framework. Exercise optional model changes using `tests/test_backends.py`; document which backend, version, device, and weights were tested. Random weights validate plumbing, not classification accuracy.

Preserve the `zmq_inference` API unless a breaking change is documented in the changelog. Add a regression test for behavior changes, update affected examples, and describe user-visible outcomes in pull requests. Do not commit weights, tokens, `.env`, or generated build artifacts.

See [architecture](docs/architecture.md) for threading and delivery constraints.
