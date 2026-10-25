import pytest

from zmq_inference import ClientConfig, ServerConfig


@pytest.mark.parametrize(
    "cls,values",
    [
        (ServerConfig, {"port": -1}),
        (ServerConfig, {"port": 65536}),
        (ServerConfig, {"max_workers": 0}),
        (ServerConfig, {"top_k": 1001}),
        (ServerConfig, {"max_pending_requests": 0}),
        (ServerConfig, {"max_image_pixels": 0}),
        (ServerConfig, {"device": "invalid"}),
        (ServerConfig, {"port": True}),
        (ClientConfig, {"port": 0}),
        (ClientConfig, {"timeout_ms": 0}),
        (ClientConfig, {"max_retries": 0}),
        (ClientConfig, {"host": "*"}),
    ],
)
def test_invalid_configuration(cls, values):
    with pytest.raises(ValueError):
        cls(**values)


def test_environment_and_explicit_values(monkeypatch):
    monkeypatch.setenv("ZMQ_PORT", "5678")
    monkeypatch.setenv("ZMQ_TIMEOUT_MS", "250")
    assert ServerConfig().port == 5678
    assert ClientConfig().timeout_ms == 250
    assert ServerConfig(port=0).port == 0
    assert ClientConfig(timeout_ms=100).timeout_ms == 100
