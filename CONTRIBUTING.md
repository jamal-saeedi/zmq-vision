# Contributing

Use Python 3.10+ and install `python -m pip install -e ".[dev]"` in a virtual environment. The `src/` layout ensures tests exercise an installed package.

Before submitting a change:

```bash
ruff check .
ruff format --check .
python -m pytest -m "not torch and not tf and not llm and not vlm"
python -m build
python -m twine check dist/*
```

Test transport changes with real localhost sockets. Keep ML imports lazy so the core package remains lightweight. Exercise PyTorch or TensorFlow changes with `tests/test_backends.py`. For the GGUF backend, install `.[llm,dev]`, download the weights from the README, and run `ZMQ_VISION_GGUF="$PWD/smollm2.gguf" python -m pytest tests/test_llm.py -m llm`. Include the runtime version, device, and weights when reporting results.

For VLM changes, install `.[vlm,dev]` and run `ZMQ_VISION_VLM_MODEL=smolvlm.gguf ZMQ_VISION_VLM_PROJECTOR=projector.gguf python -m pytest -m vlm`. Check dashboard changes with `./scripts/demo-web.sh` and `tests/test_web.py`.

Preserve the `zmq_inference` API unless a breaking change is documented in the changelog. Add a regression test for behavior changes, update affected examples, and describe user-visible outcomes in pull requests. Do not commit weights, tokens, `.env`, or generated build artifacts.

See [architecture](docs/architecture.md) for threading and delivery constraints and [benchmarks](docs/benchmarks.md) for a reproducible local workload.
