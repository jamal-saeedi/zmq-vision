import json
import signal

from click.testing import CliRunner

from zmq_inference import ServerConfig
from zmq_inference.cli.main import cli
from zmq_inference.server.demo_server import DemoInferenceServer


def test_version_and_doctor():
    runner = CliRunner()
    assert "0.3.0" in runner.invoke(cli, ["--version"]).output
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert json.loads(result.output)["libzmq"]


def test_demo_json():
    result = CliRunner().invoke(cli, ["demo", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["health"] == "ok"
    assert data["sync"]["predictions"][0]["label"] == "red"
    assert data["async"]["predictions"][0]["label"] == "blue"


def test_invalid_cli_option():
    result = CliRunner().invoke(cli, ["health", "--timeout", "0"])
    assert result.exit_code == 2


def test_health_and_inference_commands(tiny_image_path):
    server = DemoInferenceServer(ServerConfig(port=0))
    server.start()
    try:
        server.wait_until_ready(3)
        runner = CliRunner()
        port = str(server.bound_port)
        health = runner.invoke(cli, ["health", "--port", port, "--json"])
        assert health.exit_code == 0, health.output
        assert json.loads(health.output)["error"] is None
        for args in ([], ["--async"]):
            result = runner.invoke(cli, ["infer", tiny_image_path, "--port", port, "--json", *args])
            assert result.exit_code == 0, result.output
            assert json.loads(result.output)["predictions"][0]["label"] == "blue"
    finally:
        server.stop()
        server.join(3)


def test_serve_bind_failure_exits_nonzero_and_restores_signals():
    server = DemoInferenceServer(ServerConfig(port=0))
    server.start()
    previous = signal.getsignal(signal.SIGTERM)
    try:
        server.wait_until_ready(3)
        result = CliRunner().invoke(cli, ["serve", "--port", str(server.bound_port)])
        assert result.exit_code == 1
        assert "in use" in result.output
        assert signal.getsignal(signal.SIGTERM) == previous
    finally:
        server.stop()
        server.join(3)


def test_health_does_not_report_server_errors_as_healthy():
    from zmq_inference import InferenceResponse

    class Unhealthy(DemoInferenceServer):
        def process_request(self, raw):
            response = super().process_request(raw)
            return InferenceResponse(response.request_id, [], "", 0, "unavailable")

    server = Unhealthy(ServerConfig(port=0))
    server.start()
    try:
        server.wait_until_ready(3)
        result = CliRunner().invoke(cli, ["health", "--port", str(server.bound_port)])
        assert result.exit_code == 1
        assert "unavailable" in result.output
        assert "OK" not in result.output
    finally:
        server.stop()
        server.join(3)
