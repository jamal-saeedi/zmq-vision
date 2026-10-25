"""A deterministic image-statistics backend for exercising the full transport."""

from PIL import Image, ImageStat

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.protocol import Prediction
from zmq_inference.server.base import BaseInferenceServer


class DemoInferenceServer(BaseInferenceServer):
    """Rank mean RGB channel intensity. This is a demo, not a trained model."""

    def __init__(self, config: ServerConfig | None = None) -> None:
        super().__init__(config or ServerConfig())

    def load_model(self) -> None:
        pass

    def preprocess(self, img: Image.Image) -> list[float]:
        return ImageStat.Stat(img).mean

    def predict(self, tensor: list[float]) -> list[Prediction]:
        total = sum(tensor)
        scores = [v / total if total else 1 / 3 for v in tensor]
        return sorted(
            [
                Prediction(label, score)
                for label, score in zip(("red", "green", "blue"), scores, strict=True)
            ],
            key=lambda p: p.confidence,
            reverse=True,
        )[: self.config.top_k]
