"""Optional real-model TCP smoke tests; random weights need no downloads."""

import os

import pytest
from PIL import Image

from zmq_inference import ClientConfig, ServerConfig
from zmq_inference.client.sync_client import SyncClient
from zmq_inference.server.tf_server import TFInferenceServer
from zmq_inference.server.torch_server import TorchInferenceServer


@pytest.mark.torch
@pytest.mark.parametrize("pretrained", [False, True])
def test_torch_real_model(pretrained):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    torch.set_num_threads(2)
    _exercise(TorchInferenceServer, pretrained)


@pytest.mark.tf
@pytest.mark.parametrize("pretrained", [False, True])
def test_tensorflow_real_model(pretrained):
    pytest.importorskip("tensorflow")
    _exercise(TFInferenceServer, pretrained)


def _exercise(cls, pretrained):
    if pretrained and os.environ.get("ZMQ_VISION_TEST_PRETRAINED") != "1":
        pytest.skip("set ZMQ_VISION_TEST_PRETRAINED=1 to download and test real pretrained weights")
    server = cls(ServerConfig(port=0, device="cpu", top_k=5, pretrained=pretrained))
    server.start()
    try:
        server.wait_until_ready(180)
        with SyncClient(ClientConfig(port=server.bound_port, timeout_ms=30000)) as client:
            assert client.ping().ok
            response = client.infer(Image.new("RGB", (80, 60), (180, 30, 40)))
            assert response.ok, response.error
            assert len(response.predictions) == 5
            assert all(
                p.label != "unknown" and 0 <= p.confidence <= 1 for p in response.predictions
            )
            scores = [p.confidence for p in response.predictions]
            assert scores == sorted(scores, reverse=True)
    finally:
        server.stop()
        server.join(30)
        assert not server.is_alive()
