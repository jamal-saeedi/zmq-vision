"""Run with python examples/custom_backend.py; send images using zmq-vision infer."""

from PIL import Image, ImageStat

from zmq_inference import Prediction, ServerConfig
from zmq_inference.server import BaseInferenceServer


class BrightnessServer(BaseInferenceServer):
    def load_model(self) -> None:
        # Load your model here. No transport code belongs in this method.
        pass

    def preprocess(self, img: Image.Image) -> float:
        return ImageStat.Stat(img.convert("L")).mean[0] / 255

    def predict(self, tensor: float) -> list[Prediction]:
        return [Prediction("bright" if tensor >= 0.5 else "dark", max(tensor, 1 - tensor))]


if __name__ == "__main__":
    server = BrightnessServer(ServerConfig())
    server.start()
    try:
        server.wait_until_ready(10)
        print(f"Brightness backend ready on {server.config.bind_address}")
        server.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        server.join()
