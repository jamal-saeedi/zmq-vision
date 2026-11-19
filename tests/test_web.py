import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from zmq_inference import ServerConfig
from zmq_inference.server.demo_server import DemoInferenceServer
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
        assert json.load(response)["error"] is None
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
