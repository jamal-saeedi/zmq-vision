"""TensorFlow/Keras inference server for ImageNet classification."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import Prediction
from zmq_inference.server.base import BaseInferenceServer

logger = get_logger("server.tf")


class TFInferenceServer(BaseInferenceServer):
    """TensorFlow/Keras-backed inference server for ImageNet classification.

    Uses ``keras.applications.MobileNet`` by default. The model is loaded
    once in :meth:`load_model` and kept in memory for the server lifetime.
    Inference calls the model directly with ``training=False``.

    Args:
        config: Server configuration. ``config.model_name`` is informational
                for TF (the Keras model is fixed to MobileNet by default).

    Example::

        from zmq_inference import ServerConfig
        from zmq_inference.server.tf_server import TFInferenceServer

        server = TFInferenceServer(ServerConfig(port=5576))
        server.start()
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        super().__init__(config or ServerConfig())
        self._model = None
        self._preprocess_input = None
        self._labels: dict = {}

    # ------------------------------------------------------------------
    # BaseInferenceServer interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the Keras MobileNet model and configure GPU memory growth."""
        try:
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet import (  # type: ignore[import]
                MobileNet,
                preprocess_input,
            )
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Install with: pip install 'zmq-vision[tf]'"
            ) from exc

        if self.config.device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif self.config.device == "cuda" and not tf.config.list_physical_devices("GPU"):
            raise ValueError("CUDA requested but TensorFlow found no GPU")

        # Enable memory growth before initializing devices.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # Must be set before GPUs are initialised.
            logger.info("GPU(s) detected: %d device(s), memory growth enabled.", len(gpus))
        else:
            logger.info("No GPU detected. Running on CPU.")

        # Limit inter/intra-op parallelism to the available CPU count.
        max_workers = min(os.cpu_count() or 1, 4)
        try:
            tf.config.threading.set_inter_op_parallelism_threads(max_workers)
            tf.config.threading.set_intra_op_parallelism_threads(max_workers)
        except RuntimeError:
            logger.debug("TensorFlow runtime was already initialized; keeping its thread settings")

        self._model = MobileNet(weights="imagenet" if self.config.pretrained else None)
        self._preprocess_input = preprocess_input
        from zmq_inference.utils.labels import load_imagenet_labels

        self._labels = load_imagenet_labels()

        # Warm up the model with a dummy forward pass.
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self._model(dummy, training=False)
        logger.info("Keras MobileNet loaded and warm-up complete.")

    def preprocess(self, img: Image.Image) -> Any:
        """Resize and normalise a PIL image for MobileNet.

        Args:
            img: Input PIL image.

        Returns:
            A ``(1, 224, 224, 3)`` float32 NumPy array.
        """
        import numpy as np

        img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img_rgb, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return self._preprocess_input(arr)  # type: ignore[misc]

    def predict(self, tensor: Any) -> list[Prediction]:
        """Run inference and return top-k decoded ImageNet predictions.

        Args:
            tensor: Preprocessed float32 NumPy array.

        Returns:
            List of :class:`Prediction` objects sorted by confidence (desc).
        """
        import numpy as np

        raw_preds = np.asarray(self._model(tensor, training=False))[0]
        top_ids = np.argsort(raw_preds)[::-1][: self.config.top_k]
        return [
            Prediction(
                label=self._labels.get(str(int(index)), "unknown"),
                confidence=round(float(raw_preds[index]), 6),
            )
            for index in top_ids
        ]
