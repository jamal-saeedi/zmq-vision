import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from click.testing import CliRunner

from zmq_inference import InferenceRequest, ServerConfig
from zmq_inference.cli.main import cli
from zmq_inference.server.demo_server import DemoInferenceServer
from zmq_inference.server.llm_server import LLMInferenceServer
from zmq_inference.server.vlm_server import VLMInferenceServer
from zmq_inference.web import DemoWebServer


@pytest.fixture
def web_url(serve):
    config = serve(DemoInferenceServer(ServerConfig(host="127.0.0.1", port=0)))
    with DemoWebServer(0, config) as web:
        worker = Thread(target=web.serve_forever)
        worker.start()
        try:
            yield web.url
        finally:
            web.shutdown()
            worker.join(5)
            assert not worker.is_alive()


def test_web_serves_assets_and_real_tcp_inference(web_url, tiny_image_b64):
    with urlopen(web_url, timeout=3) as response:
        assert "Model workbench" in response.read().decode()
        assert "connect-src 'self'" in response.headers["Content-Security-Policy"]
    for asset in ("app.js", "app.css"):
        with urlopen(f"{web_url}/{asset}", timeout=3) as response:
            assert response.read()
    with urlopen(f"{web_url}/api/health", timeout=3) as response:
        health = json.load(response)
        assert health["error"] is None
        assert health["operations"] == ["infer"]
    request = Request(
        f"{web_url}/api/request",
        data=json.dumps({"operation": "infer", "payload": tiny_image_b64}).encode(),
        headers={"Content-Type": "application/json", "Origin": web_url},
    )
    with urlopen(request, timeout=3) as response:
        data = json.load(response)
        assert data["predictions"][0]["label"] == "blue"
        assert data["request_id"] and "image" not in data


def test_web_rejects_cross_origin_and_wrong_operation(web_url):
    request = Request(f"{web_url}/api/health", headers={"Origin": "https://example.test"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=3)
    assert error.value.code == 403

    request = Request(
        f"{web_url}/api/request",
        data=b'{"operation":"generate","prompt":"Hello"}',
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=3)
    assert error.value.code == 422
    assert "does not support text generation" in json.load(error.value)["error"]


def test_ping_capabilities_match_each_generation_backend():
    config = ServerConfig()
    llm = LLMInferenceServer(config, model_path="ignored.gguf")
    vlm = VLMInferenceServer(config, model_path="ignored.gguf", projector_path="ignored.gguf")
    ping = InferenceRequest(operation="ping")
    for server, operations in ((llm, ["generate"]), (vlm, ["generate", "describe"])):
        assert "operations" not in server.process_request(ping.to_json().encode()).to_dict()
        ping_with_operations = InferenceRequest(
            operation="ping", metadata={"include_operations": True}
        )
        response = server.process_request(ping_with_operations.to_json().encode())
        assert response.operations == operations


def test_web_vlm_requires_files_and_rejects_conflicting_modes():
    runner = CliRunner()
    missing = runner.invoke(cli, ["web", "--vlm"])
    assert missing.exit_code == 2
    assert "--vlm requires both --model-path and --projector-path" in missing.output

    conflict = runner.invoke(cli, ["web", "--demo", "--vlm"])
    assert conflict.exit_code == 2
    assert "use only one" in conflict.output
