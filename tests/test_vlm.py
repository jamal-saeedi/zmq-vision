import json
import os

import pytest
from click.testing import CliRunner
from PIL import Image

from zmq_inference import Generation, InferenceRequest, ServerConfig
from zmq_inference.cli.main import cli
from zmq_inference.client import SyncClient
from zmq_inference.server.demo_server import DemoInferenceServer
from zmq_inference.server.vlm_server import VLMInferenceServer
from zmq_inference.utils.image import decode_image, encode_image


def test_describe_wire_format_and_requires_image(tiny_image_b64):
    request = InferenceRequest(
        operation="describe", payload=tiny_image_b64, prompt="What is shown?", max_tokens=5
    )
    assert InferenceRequest.from_json(request.to_json()) == request
    with pytest.raises(ValueError, match="nonempty payload"):
        InferenceRequest(operation="describe", prompt="What is shown?")


def test_vlm_decodes_bounds_and_resizes_images(monkeypatch):
    server = VLMInferenceServer(
        ServerConfig(max_image_pixels=640 * 640),
        model_path="unused.gguf",
        projector_path="unused.gguf",
    )

    def fake_complete(prompt, max_tokens, image_b64):
        with decode_image(image_b64) as image:
            assert image.size == (512, 256)
        return Generation(prompt, 10, max_tokens, "length")

    monkeypatch.setattr(server, "_complete", fake_complete)
    with Image.new("RGB", (640, 320)) as image:
        request = InferenceRequest(
            operation="describe", payload=encode_image(image), prompt="Hello", max_tokens=4
        )
    good = server.process_request(request.to_json().encode())
    assert good.generation == Generation("Hello", 10, 4, "length")
    for payload in ("invalid", encode_image(Image.new("RGB", (641, 640)))):
        request.payload = payload
        failed = server.process_request(request.to_json().encode())
        assert not failed.ok and failed.request_id == request.request_id


def test_generate_image_requires_correct_backend(serve, tiny_image_path):
    config = serve(DemoInferenceServer(ServerConfig(host="127.0.0.1", port=0)))
    result = CliRunner().invoke(
        cli,
        ["generate", "What is shown?", "--image", tiny_image_path, "--port", str(config.port)],
    )
    assert result.exit_code == 1
    assert "does not support image-conditioned generation" in result.output

    missing = CliRunner().invoke(cli, ["serve", "--backend", "vlm", "--model-path", "model.gguf"])
    assert missing.exit_code == 2
    assert "--projector-path" in missing.output


@pytest.mark.vlm
def test_real_smolvlm_uses_the_image(serve, tmp_path):
    model = os.environ.get("ZMQ_VISION_VLM_MODEL")
    projector = os.environ.get("ZMQ_VISION_VLM_PROJECTOR")
    if not model or not projector:
        pytest.skip("set both VLM weight paths to the documented SmolVLM files")
    config = serve(
        VLMInferenceServer(
            ServerConfig(host="127.0.0.1", port=0, device="cpu"),
            model_path=model,
            projector_path=projector,
        )
    )
    prompt = "What is the dominant color in this image? Give one word."
    with SyncClient(config) as client, Image.new("RGB", (224, 224), "red") as red:
        assert client.ping().ok
        first = client.generate(prompt, image=red, max_tokens=16)
        assert first.ok, first.error
        assert "red" in first.generation.text.lower()

    path = tmp_path / "blue.png"
    with Image.new("RGB", (224, 224), "blue") as blue:
        blue.save(path)
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            prompt,
            "--image",
            str(path),
            "--async",
            "--json",
            "--max-tokens",
            "16",
            "--port",
            str(config.port),
        ],
    )
    assert result.exit_code == 0, result.output
    second = json.loads(result.output)
    assert first.request_id != second["request_id"]
    assert "blue" in second["generation"]["text"].lower()
    assert 1 <= second["generation"]["completion_tokens"] <= 16

    with SyncClient(config) as client:
        text = client.generate("What is 2 plus 2?", max_tokens=16)
        assert text.ok, text.error
        assert "4" in text.generation.text
        assert text.generation.prompt_tokens < first.generation.prompt_tokens
