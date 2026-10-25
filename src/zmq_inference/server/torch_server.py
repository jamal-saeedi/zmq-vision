"""PyTorch inference server using timm for model loading."""

from __future__ import annotations

from typing import Any

from PIL import Image

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import Prediction
from zmq_inference.server.base import BaseInferenceServer
from zmq_inference.utils.labels import get_label, load_imagenet_labels

logger = get_logger("server.torch")


class TorchInferenceServer(BaseInferenceServer):
    """PyTorch/timm-backed inference server for ImageNet classification.

    The model is loaded from the ``timm`` model zoo based on
    :attr:`~zmq_inference.core.config.ServerConfig.model_name`. Use an ImageNet-1k classifier
    with a standard 1000-class output and matching class order.

    Args:
        config: Server configuration. ``config.device`` may be ``"auto"``,
                ``"cpu"``, or ``"cuda"``.

    Example::

        from zmq_inference import ServerConfig
        from zmq_inference.server.torch_server import TorchInferenceServer

        server = TorchInferenceServer(ServerConfig(port=5576, model_name="mobilenetv2_100"))
        server.start()
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        super().__init__(config or ServerConfig())
        self._model = None
        self._transform = None
        self._labels: dict = {}
        self._device = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        cfg = self.config.device.lower()
        if cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cfg)

    # ------------------------------------------------------------------
    # BaseInferenceServer interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the timm model and ImageNet labels into memory."""
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise ImportError(
                "PyTorch dependencies are not installed. "
                "Install with: pip install 'zmq-vision[torch]'"
            ) from exc

        self._device = self._resolve_device()
        logger.info("Using device: %s", self._device)

        self._model = timm.create_model(self.config.model_name, pretrained=self.config.pretrained)
        if self._model.num_classes != 1000:
            raise ValueError("This backend requires a 1000-class ImageNet model")
        self._model.eval()
        self._model.to(self._device)
        logger.info("Loaded timm model: %s", self.config.model_name)

        self._transform = create_transform(
            **resolve_model_data_config(self._model), is_training=False
        )

        self._labels = load_imagenet_labels()

    def preprocess(self, img: Image.Image) -> Any:
        """Apply ImageNet normalisation transforms to a PIL image.

        Args:
            img: Input RGB PIL image.

        Returns:
            A batched ``torch.Tensor`` on :attr:`_device`.
        """
        img_rgb = img.convert("RGB")
        tensor = self._transform(img_rgb)  # type: ignore[misc]
        return tensor.unsqueeze(0).to(self._device)  # type: ignore[union-attr]

    def predict(self, tensor: Any) -> list[Prediction]:
        """Run a forward pass and return top-k predictions.

        Args:
            tensor: Batched tensor from :meth:`preprocess`.

        Returns:
            List of :class:`Prediction` objects sorted by confidence (desc).
        """
        import torch

        with torch.inference_mode():
            outputs = self._model(tensor)  # type: ignore[misc]
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top_probs, top_ids = torch.topk(probs, self.config.top_k)
        predictions = [
            Prediction(
                label=get_label(cat_id.item(), self._labels),
                confidence=round(float(prob.item()), 6),
            )
            for prob, cat_id in zip(top_probs, top_ids, strict=True)
        ]
        return predictions
