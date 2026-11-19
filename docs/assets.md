# Visual and data assets

`docs/assets/banner.svg` is an original vector graphic created for this project. `docs/assets/demo.svg` renders terminal output from a real local TCP demo, captured in `demo-output.txt`. Regenerate both with `python scripts/render_assets.py` after installing the package.

`docs/assets/workbench.png` captures the local dashboard running an actual SmolVLM request against a generated sample image.

`examples/sample.png` and `src/zmq_inference/data/imagenet_classes.json` are retained from the original repository. Their original upstream provenance was not recorded. The generated-image demo avoids relying on the sample image. Do not assume that the MIT software license establishes rights to third-party images or model weights.
