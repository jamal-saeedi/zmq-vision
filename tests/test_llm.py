import json
import os

import pytest
from click.testing import CliRunner

from zmq_inference import (
    ClientConfig,
    Generation,
    InferenceRequest,
    InferenceResponse,
    ServerConfig,
)
from zmq_inference.cli.main import cli
from zmq_inference.client import SyncClient
from zmq_inference.client.async_client import AsyncClient
from zmq_inference.server.base import BaseModelServer
from zmq_inference.server.demo_server import DemoInferenceServer
from zmq_inference.server.llm_server import LLMInferenceServer


class TextStub(BaseModelServer):
    def load_model(self):
        pass

    def _generate(self, prompt, max_tokens):
        return Generation(f"Answer to {prompt}", 8, max_tokens, "length")


@pytest.fixture
def serve():
    servers = []

    def start(server):
        servers.append(server)
        server.start()
        server.wait_until_ready(30)
        return ClientConfig(host="127.0.0.1", port=server.bound_port)

    yield start
    for server in servers:
        server.stop()
        server.join(5)
        assert not server.is_alive()


def test_wire_shapes_remain_compatible():
    request = InferenceRequest(payload="image", request_id="vision")
    assert request.to_dict() == {"payload": "image", "request_id": "vision", "metadata": {}}
    image = InferenceResponse("vision", [], "", 0)
    assert image.to_dict() == {
        "request_id": "vision",
        "predictions": [],
        "image": "",
        "elapsed_ms": 0,
        "error": None,
    }
    text = InferenceRequest(operation="generate", prompt="Café?", max_tokens=5)
    restored = InferenceRequest.from_json(text.to_json())
    assert restored == text
    response = TextStub(ServerConfig()).process_request(text.to_json().encode())
    assert response.generation == Generation("Answer to Café?", 8, 5, "length")
    assert InferenceResponse.from_json(response.to_json()) == response


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": "   "},
        {"prompt": 12},
        {"prompt": "é" * 4097},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_tokens": 257},
        {"payload": "image"},
    ],
)
def test_server_rejects_invalid_generation_with_correlated_error(overrides):
    data = {"operation": "generate", "request_id": "bad", "prompt": "Hello"} | overrides
    response = TextStub(ServerConfig()).process_request(json.dumps(data).encode())
    assert not response.ok
    assert response.request_id == "bad"
    assert response.generation is None


def test_text_generation_sync_and_cli(serve, tiny_pil_image):
    config = serve(TextStub(ServerConfig(host="127.0.0.1", port=0)))
    with SyncClient(config) as client:
        assert client.ping().ok
        assert not client.infer(tiny_pil_image).ok
        result = client.generate("Hello", max_tokens=4)
        assert result.generation == Generation("Answer to Hello", 8, 4, "length")
    for mode in ([], ["--async"]):
        result = CliRunner().invoke(
            cli,
            ["generate", "--port", str(config.port), "--max-tokens", "4", "--json", *mode, "Hello"],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["generation"]["completion_tokens"] == 4


async def test_text_generation_async_reuse_and_wrong_backend(serve):
    config = serve(TextStub(ServerConfig(host="127.0.0.1", port=0)))
    async with AsyncClient(config) as client:
        first = await client.generate("First")
        second = await client.generate("Second")
    assert first.request_id != second.request_id
    assert [first.generation.text, second.generation.text] == [
        "Answer to First",
        "Answer to Second",
    ]

    config = serve(DemoInferenceServer(ServerConfig(host="127.0.0.1", port=0)))
    async with AsyncClient(config) as client:
        response = await client.generate("Hello")
        assert response.error == "this backend does not support text generation"
        assert (await client.ping()).ok


def test_llm_cli_requires_explicit_weights():
    result = CliRunner().invoke(cli, ["serve", "--backend", "llm"])
    assert result.exit_code == 2
    assert "--model-path" in result.output


@pytest.mark.llm
def test_real_gguf_over_tcp(serve):
    weights = os.environ.get("ZMQ_VISION_GGUF")
    if not weights:
        pytest.skip("set ZMQ_VISION_GGUF to the documented SmolLM2 weights")

    config = serve(
        LLMInferenceServer(
            ServerConfig(host="127.0.0.1", port=0, device="cpu"),
            model_path=weights,
            context_tokens=256,
        )
    )
    with SyncClient(config) as client:
        assert client.ping().ok
        too_long = client.generate("one two " * 200, max_tokens=8)
        assert too_long.error == "prompt and max_tokens exceed the model context"
        result = client.generate("Name three primary colors.", max_tokens=24)
        assert result.ok, result.error
        assert result.generation.text.strip()
        assert 1 <= result.generation.completion_tokens <= 24
        assert result.generation.prompt_tokens > 0
        assert result.generation.finish_reason in {"stop", "length"}
