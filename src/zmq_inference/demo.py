"""Download-free end-to-end demo with real TCP sockets and generated images."""

from __future__ import annotations

import asyncio

from PIL import Image

from zmq_inference.client.async_client import AsyncClient
from zmq_inference.client.sync_client import SyncClient
from zmq_inference.core.config import ClientConfig, ServerConfig
from zmq_inference.server.demo_server import DemoInferenceServer


def run_demo() -> dict:
    server = DemoInferenceServer(ServerConfig(host="127.0.0.1", port=0))
    server.start()
    try:
        server.wait_until_ready(timeout=10)
        config = ClientConfig(host="127.0.0.1", port=server.bound_port, timeout_ms=3000)
        with SyncClient(config) as client:
            ping = client.ping()
            sync = client.infer(Image.new("RGB", (64, 64), (240, 30, 20)))

        async def async_infer():
            async with AsyncClient(config) as client:
                return await client.infer(Image.new("RGB", (64, 64), (20, 30, 240)))

        asynchronous = asyncio.run(async_infer())
        if not all(response.ok for response in (ping, sync, asynchronous)):
            raise RuntimeError("Demo server returned an error")
        if sync.predictions[0].label != "red" or asynchronous.predictions[0].label != "blue":
            raise RuntimeError("Demo predictions did not match generated image colors")
        return {
            "backend": "demo (RGB statistics; no trained model)",
            "transport": "ZeroMQ DEALER / ROUTER over local TCP",
            "health": "ok",
            "sync": sync.to_dict() | {"image": "<omitted>"},
            "async": asynchronous.to_dict() | {"image": "<omitted>"},
        }
    finally:
        server.stop()
        server.join(timeout=10)
        if server.is_alive():
            raise RuntimeError("Demo server failed to stop")
