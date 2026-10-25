# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-10-25

### Added

- Published `zmq-vision` 0.3.0 on PyPI with matching GitHub release artifacts.
- ZMQ Vision branding and `zmq-vision` distribution/CLI, retaining `zmq_inference` imports.
- Download-free TCP demo, readiness protocol, environment diagnostics, and JSON CLI output.
- Bounded model workers and admission, validation limits, overload responses, and startup failure reporting.
- Real TCP, concurrent client, timeout, cancellation, shutdown, CLI, and optional real-model tests.
- CI across supported Python versions, distribution checks, tagged GitHub releases, and manual OIDC PyPI publishing.
- Architecture, migration, contribution, and release guides; captured demo visualization; missing MIT license file.

### Changed

- Moved library to `src/`, sample to `examples/`, and original scripts to `examples/legacy/`.
- Consolidated metadata into `pyproject.toml`; removed obsolete GPU requirements and duplicate label data.
- Updated dependency ranges to current supported releases, with ML frameworks kept optional.
- Included the Windows-only Tornado selector bridge required by PyZMQ asyncio.
- Applied model-specific timm transforms and direct TensorFlow/Keras inference.
- Made localhost binding and one client attempt the defaults; health no longer performs inference.
- Replaced request threads and the in-process proxy with bounded futures and one socket-owning thread.
- Reset client sockets after timeout/cancellation and correlate every response.

See [migration notes](docs/migration.md) for API and default changes.

---

## [0.2.0] - 2026-05-24

### Changed

- **Complete package overhaul**: flat scripts migrated to the `zmq_inference` Python package.
- **Abstract base classes** for server (`BaseInferenceServer`) and client (`BaseInferenceClient`) with clean extension points.
- **Dataclass-based configuration** (`ServerConfig`, `ClientConfig`) with full environment-variable support.
- **Typed protocol layer** (`InferenceRequest`, `InferenceResponse`, `Prediction`) with JSON serialisation helpers.
- **Click CLI** entry point (`zmq-inference serve | infer | health`).
- **Full test suite** (`tests/`) with pytest, pytest-asyncio, and pytest-cov.
- **pyproject.toml** build system with optional extras (`torch`, `tf`, `dev`).
- **Async client** (`AsyncClient`) using `zmq.asyncio`.
- **Structured logging** via `zmq_inference.core.logging`.
- **ImageNet labels** bundled as package data under `zmq_inference/data/`.
- **Makefile** for common development tasks.
- Updated dependencies to modern versions (pyzmq>=25, Pillow>=9, numpy>=1.23).
- TensorFlow server modernised: removed `tf.compat.v1` session usage; uses `model.predict` with `tf.function` warm-up.

### Added

- `zmq_inference/server/base.py` — abstract `BaseInferenceServer` (threading.Thread + ZMQ loop).
- `zmq_inference/server/torch_server.py` — PyTorch/timm implementation.
- `zmq_inference/server/tf_server.py` — TensorFlow/Keras implementation (TF2 style).
- `zmq_inference/client/base.py` — abstract `BaseInferenceClient`.
- `zmq_inference/client/sync_client.py` — synchronous client with retry logic.
- `zmq_inference/client/async_client.py` — asyncio client.
- `zmq_inference/core/config.py` — `ServerConfig` and `ClientConfig` dataclasses.
- `zmq_inference/core/protocol.py` — request/response dataclasses.
- `zmq_inference/core/logging.py` — structured logging helpers.
- `zmq_inference/utils/image.py` — `encode_image`, `decode_image`, `preprocess_for_display`.
- `zmq_inference/utils/labels.py` — `load_imagenet_labels`, `get_label` (uses `importlib.resources`).
- `zmq_inference/cli/main.py` — Click CLI: `serve`, `infer`, `health` commands.
- `.env.example` — environment variable reference.
- `.gitignore` — comprehensive Python + ML project ignore rules.

---

## [0.1.0] - 2024-06-12

### Added

- Initial ZMQ inference server supporting PyTorch (timm) and TensorFlow (Keras) backends.
- Synchronous polling client (`client_loop.py`).
- Async client using `asyncio.to_thread` (`client_async.py`).
- `imagenet_classes.json` label file.
- `requirements.txt` with pinned dependencies.
