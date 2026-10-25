"""Regression tests for request validation and server lifecycle."""

import json

import pytest

from zmq_inference import ServerConfig
from zmq_inference.core.protocol import InferenceRequest
from zmq_inference.server.demo_server import DemoInferenceServer


def test_process_request(tiny_image_b64):
    server = DemoInferenceServer()
    result = server.process_request(
        InferenceRequest(tiny_image_b64, "preserved").to_json().encode()
    )
    assert result.ok
    assert result.request_id == "preserved"
    assert result.predictions[0].label == "blue"
    assert result.image
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"[]",
        b"null",
        b"1",
        b'"text"',
        b"\xff",
        b"{}",
        b'{"payload":123}',
        b'{"payload":"not base64!!!"}',
    ],
)
def test_invalid_requests_are_error_responses(raw):
    assert not DemoInferenceServer().process_request(raw).ok


def test_errors_keep_request_id():
    raw = json.dumps({"request_id": "bad-image", "payload": "not base64!!!"}).encode()
    response = DemoInferenceServer().process_request(raw)
    assert not response.ok
    assert response.request_id == "bad-image"


def test_message_size_limit():
    server = DemoInferenceServer(ServerConfig(max_message_bytes=20))
    assert "size limit" in server.process_request(b" " * 21).error


def test_pixel_limit(tiny_image_b64):
    server = DemoInferenceServer(ServerConfig(max_image_pixels=1))
    raw = InferenceRequest(tiny_image_b64).to_json().encode()
    assert "pixel limit" in server.process_request(raw).error


def test_ping_skips_model():
    class NoInference(DemoInferenceServer):
        def predict(self, tensor):
            raise AssertionError("ping must not run inference")

    response = NoInference().process_request(InferenceRequest(operation="ping").to_json().encode())
    assert response.ok
    assert response.predictions == []
    assert response.image == ""


def test_stop_flag():
    server = DemoInferenceServer()
    assert not server.stopped()
    server.stop()
    assert server.stopped()
