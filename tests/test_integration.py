"""Real TCP integration tests; no mock sockets or model downloads."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq
from PIL import Image

from zmq_inference import ClientConfig, ServerConfig
from zmq_inference.client.async_client import AsyncClient
from zmq_inference.client.sync_client import SyncClient
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse
from zmq_inference.server.demo_server import DemoInferenceServer
from zmq_inference.utils.image import encode_image


@pytest.fixture
def server_factory():
    servers = []

    def make(cls=DemoInferenceServer, **kwargs):
        server = cls(ServerConfig(host="127.0.0.1", port=0, poll_timeout_ms=5, **kwargs))
        servers.append(server)
        server.start()
        server.wait_until_ready(10)
        return server

    yield make
    for server in servers:
        server.stop()
    for server in servers:
        server.join(5)
        assert not server.is_alive(), "server leaked a thread on shutdown"
        assert server.error is None


def client_config(server, **kwargs):
    return ClientConfig(host="127.0.0.1", port=server.bound_port, timeout_ms=2000, **kwargs)


def test_round_trip_and_idempotent_connect(server_factory, tiny_pil_image):
    server = server_factory()
    with SyncClient(client_config(server)) as client:
        original_socket = client._socket
        client.connect()
        assert client._socket is original_socket
        assert client.ping().ok
        result = client.infer(tiny_pil_image)
        assert result.ok
        assert result.predictions[0].label == "blue"
        assert result.image
    client.disconnect()


def test_parallel_clients(server_factory):
    server = server_factory()

    def infer(index):
        with SyncClient(client_config(server)) as client:
            response = client.infer(Image.new("RGB", (8, 8), (200, 10, 10)))
            assert response.ok
            return response.request_id

    with ThreadPoolExecutor(max_workers=6) as executor:
        ids = list(executor.map(infer, range(12)))
    assert len(set(ids)) == 12


def test_sync_thread_ownership(server_factory):
    with SyncClient(client_config(server_factory())) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(RuntimeError, match="connecting thread"):
                executor.submit(client.ping).result(timeout=2)
        assert client.ping().ok


def test_bad_frames_do_not_crash_server(server_factory, tiny_image_b64):
    server = server_factory()
    with zmq.Context() as context:
        with context.socket(zmq.DEALER) as socket:
            socket.linger = 0
            socket.connect(client_config(server).server_address)
            for frames in ([b"{"], [b"[]"], [b"null"], [b"one", b"two"]):
                socket.send_multipart(frames)
                assert socket.poll(2000)
                assert not InferenceResponse.from_json(socket.recv_string()).ok
            socket.send_string(InferenceRequest(tiny_image_b64).to_json())
            assert socket.poll(2000)
            assert InferenceResponse.from_json(socket.recv_string()).ok
    assert server.is_alive()


def test_bounded_admission_and_worker_count(server_factory, tiny_image_b64):
    entered, release = threading.Event(), threading.Event()

    class BlockedServer(DemoInferenceServer):
        def predict(self, tensor):
            entered.set()
            assert release.wait(3)
            return super().predict(tensor)

    server = server_factory(BlockedServer, max_pending_requests=1, max_workers=1)
    try:
        with zmq.Context() as context:
            with context.socket(zmq.DEALER) as socket:
                socket.linger = 0
                socket.connect(client_config(server).server_address)
                socket.send_string(InferenceRequest(tiny_image_b64, "first").to_json())
                assert entered.wait(2)
                socket.send_string(InferenceRequest(tiny_image_b64, "second").to_json())
                assert socket.poll(2000)
                busy = InferenceResponse.from_json(socket.recv_string())
                assert busy.request_id == "second"
                assert "busy" in busy.error
                release.set()
                assert socket.poll(2000)
                first = InferenceResponse.from_json(socket.recv_string())
                assert first.ok and first.request_id == "first"
    finally:
        release.set()


class SlowServer(DemoInferenceServer):
    def predict(self, tensor):
        time.sleep(0.12)
        return super().predict(tensor)


def test_timeout_does_not_poison_next_request(server_factory, tiny_pil_image):
    server = server_factory(SlowServer)
    config = ClientConfig(port=server.bound_port, timeout_ms=35)
    with SyncClient(config) as client:
        with pytest.raises(TimeoutError):
            client.infer(tiny_pil_image)
        config.timeout_ms = 2000
        response = client.infer(Image.new("RGB", (4, 4), (230, 10, 10)))
        assert response.ok and response.predictions[0].label == "red"


def test_retry_delayed_response_uses_new_identity(server_factory, tiny_pil_image):
    class InitiallySlow(DemoInferenceServer):
        calls = 0

        def predict(self, tensor):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.18)
            return super().predict(tensor)

    server = server_factory(InitiallySlow)
    config = ClientConfig(port=server.bound_port, timeout_ms=130, max_retries=3)
    with SyncClient(config) as client:
        response = client.infer(tiny_pil_image)
        assert response.ok
        assert client.ping().ok
    assert server.calls >= 2


async def test_async_concurrent_calls_are_correlated(server_factory):
    server = server_factory()
    async with AsyncClient(client_config(server)) as client:
        await client.connect()
        colors = [(230, 10, 10), (10, 230, 10), (10, 10, 230)]
        responses = await asyncio.gather(
            *[client.infer(Image.new("RGB", (8, 8), color)) for color in colors]
        )
        assert [r.predictions[0].label for r in responses] == ["red", "green", "blue"]
        assert len({r.request_id for r in responses}) == 3
        assert (await client.ping()).ok


async def test_async_cancellation_recovers(server_factory):
    entered = threading.Event()

    class ObservableSlow(SlowServer):
        def predict(self, tensor):
            entered.set()
            return super().predict(tensor)

    server = server_factory(ObservableSlow)
    async with AsyncClient(client_config(server)) as client:
        task = asyncio.create_task(client.infer(Image.new("RGB", (4, 4), "blue")))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        response = await client.infer(Image.new("RGB", (4, 4), "red"))
        assert response.ok and response.predictions[0].label == "red"


async def test_async_timeout_recovers(server_factory, tiny_pil_image):
    server = server_factory(SlowServer)
    config = ClientConfig(port=server.bound_port, timeout_ms=35)
    async with AsyncClient(config) as client:
        with pytest.raises(TimeoutError):
            await client.infer(tiny_pil_image)
        config.timeout_ms = 2000
        result = await client.infer(Image.new("RGB", (4, 4), "red"))
        assert result.ok and result.predictions[0].label == "red"


def test_startup_failure_is_propagated():
    class BrokenServer(DemoInferenceServer):
        def load_model(self):
            raise ValueError("bad model")

    server = BrokenServer()
    server.start()
    with pytest.raises(RuntimeError, match="bad model"):
        server.wait_until_ready(2)
    server.join(2)
    assert not server.is_alive()


def test_bind_failure_releases_resources(server_factory):
    running = server_factory()
    server = DemoInferenceServer(ServerConfig(port=running.bound_port))
    server.start()
    with pytest.raises(RuntimeError):
        server.wait_until_ready(2)
    server.join(2)
    assert server.error.errno == zmq.EADDRINUSE
    assert not server.is_alive()


def test_shutdown_waits_for_running_inference(server_factory):
    entered, release = threading.Event(), threading.Event()

    class Blocking(DemoInferenceServer):
        def predict(self, tensor):
            entered.set()
            assert release.wait(3)
            return super().predict(tensor)

    server = server_factory(Blocking)
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.linger = 0
    try:
        socket.connect(client_config(server).server_address)
        socket.send_string(InferenceRequest(encode_image(Image.new("RGB", (4, 4)))).to_json())
        assert entered.wait(2)
        server.stop()
        server.join(0.05)
        assert server.is_alive()
        release.set()
        server.join(2)
        assert not server.is_alive()
    finally:
        release.set()
        socket.close()
        context.term()


class WrongIdFirstServer(DemoInferenceServer):
    @staticmethod
    def _send(socket, identity, response):
        wrong = InferenceResponse("unrelated-request", [], "", 0)
        DemoInferenceServer._send(socket, identity, wrong)
        DemoInferenceServer._send(socket, identity, response)


def test_sync_ignores_unrelated_response_ids(server_factory, tiny_pil_image):
    server = server_factory(WrongIdFirstServer)
    with SyncClient(client_config(server)) as client:
        response = client.infer(tiny_pil_image)
        assert response.request_id != "unrelated-request"
        assert response.predictions[0].label == "blue"


async def test_async_ignores_unrelated_response_ids(server_factory, tiny_pil_image):
    server = server_factory(WrongIdFirstServer)
    async with AsyncClient(client_config(server)) as client:
        response = await client.infer(tiny_pil_image)
        assert response.request_id != "unrelated-request"
        assert response.predictions[0].label == "blue"
