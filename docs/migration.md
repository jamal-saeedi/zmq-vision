# Migrating to 0.3

The project is now **ZMQ Vision**, distribution `zmq-vision`. Install from [PyPI](https://pypi.org/project/zmq-vision/) with `python -m pip install zmq-vision`, from this repository, or from a release wheel.

Existing `zmq_inference` Python imports and `zmq-inference` CLI command remain available. `zmq-vision` is the preferred CLI name. The package moved into `src/`; editable installation is required for local development. `requirements.txt` now delegates to project metadata instead of pinning obsolete GPU packages.

## Intentional changes

- Python 3.10+ is required. Core dependencies are current PyZMQ, Pillow, and Click. NumPy is supplied by ML extras or development tools.
- The bind host defaults to `127.0.0.1`. Explicitly pass `--host 0.0.0.0` to accept trusted LAN clients.
- The CLI defaults to the download-free demo backend. Select `--backend torch` or `--backend tf` for ML inference.
- Client attempts default to one. Set `ClientConfig(max_retries=3)` or `infer --retries 3` to opt into repeated attempts.
- The server uses bounded futures instead of an unbounded thread per request. Internal `RequestHandler` and `ServerConfig.backend_endpoint` are removed.
- `health` uses the new `ping` operation. Older 0.2 servers do not support it.
- Startup failures are available as `server.error` and through `wait_until_ready()`. The CLI exits unsuccessfully on startup failure.
- `.env` files are not automatically read. Export the variables explicitly; the earlier README's automatic loading claim was incorrect.
- Original flat scripts are archived under `examples/legacy/`. Those 2024 scripts use a different protocol and are unsupported.
- `sample.png` moved to `examples/sample.png`. ImageNet label data has one bundled source under the package.

Models should produce ImageNet-1k classes for the supplied label mapping. Custom task-specific labels belong in a custom backend. The legacy distribution is not an automatic dependency of the new distribution; avoid installing both into one environment because they share an import namespace.
