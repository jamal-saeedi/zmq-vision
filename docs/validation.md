# Release 0.3.0 validation

Verified locally on 2026-10-25 with Linux / CPython 3.12.3. The installed package was exercised through real localhost TCP sockets.

| Component | Tested version |
| --- | --- |
| PyZMQ / libzmq | 27.2.0 / 4.3.5 |
| Pillow | 13.0.0 |
| Click | 8.5.0 |
| NumPy (ML and tests) | 2.5.4 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 |
| timm | 1.0.30 |
| TensorFlow / Keras | 2.21.0 / 3.15.1 |

- 98 tests passed, including pretrained and random-weight MobileNet inference through both ML backends. Coverage: 92% across 779 executable statements.
- Real TCP tests cover concurrent clients, response IDs, retries, stale replies, async cancellation, invalid frames, invalid images, admission limits, startup/bind failures, and draining running work on shutdown.
- Both distributions passed `twine check`. The wheel was installed into a fresh environment and tested from outside the checkout, including bundled labels, both CLI names, and the sync/async demo. NumPy and ML frameworks were absent from that core environment.
- Ruff lint and format checks passed. Source distribution contents, workflow YAML and embedded Python, local documentation links, and credential patterns were checked. The SVGs were rendered and visually inspected.

Pretrained checks can be reproduced with:

```bash
ZMQ_VISION_TEST_PRETRAINED=1 OMP_NUM_THREADS=2 python -m pytest --cov=zmq_inference
```

Build isolation on this machine used `python -m build --installer uv` because its system Python lacks `ensurepip`. Standard `python -m build` works with a Python installation that includes virtual-environment support.

GitHub CI independently checks the supported Python matrix and additional operating systems; local results above do not establish Windows or macOS behavior. GPU execution, classification accuracy on a benchmark, load capacity, and hard shutdown of hung native model code were not measured.
