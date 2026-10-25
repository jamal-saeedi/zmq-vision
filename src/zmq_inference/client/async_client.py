"""Async transport with serialized exchanges and cancellation-safe recovery."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import zmq
import zmq.asyncio
from PIL import Image

from zmq_inference.core.config import ClientConfig
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse
from zmq_inference.utils.image import encode_image


class AsyncClient:
    """An asyncio client scoped to one event loop.

    Concurrent infer() calls on one client are serialized to preserve response
    ownership. Use separate clients to run requests in parallel on the server.
    Retries can repeat model execution; max_retries counts total attempts.
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig()
        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._lock = asyncio.Lock()

    def _open_socket(self) -> None:
        assert self._context is not None
        socket = self._context.socket(zmq.DEALER)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt_string(zmq.IDENTITY, str(uuid.uuid4()))
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.connect(self.config.server_address)
        except Exception:
            socket.close(linger=0)
            raise
        self._socket = socket

    async def connect(self) -> None:
        async with self._lock:
            if self._socket is not None:
                return
            self._context = zmq.asyncio.Context()
            try:
                self._open_socket()
            except Exception:
                self._context.term()
                self._context = None
                raise

    async def disconnect(self) -> None:
        async with self._lock:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None

    async def _attempt(self, request: InferenceRequest) -> InferenceResponse:
        assert self._socket is not None
        await self._socket.send_json(request.to_dict())
        while True:
            response = InferenceResponse.from_json(await self._socket.recv_string())
            if response.request_id == request.request_id:
                return response

    async def _exchange(self, request: InferenceRequest) -> InferenceResponse:
        async with self._lock:
            if self._socket is None:
                raise RuntimeError("AsyncClient is not connected. Use an async context manager.")
            for _ in range(self.config.max_retries):
                try:
                    return await asyncio.wait_for(
                        self._attempt(request), self.config.timeout_ms / 1000
                    )
                except asyncio.TimeoutError:
                    self._socket.close(linger=0)
                    self._open_socket()
                except asyncio.CancelledError:
                    self._socket.close(linger=0)
                    self._open_socket()
                    raise
            raise TimeoutError(
                f"Server at {self.config.server_address} did not respond after "
                f"{self.config.max_retries} attempt(s) ({self.config.timeout_ms} ms each)."
            )

    async def infer(self, source: str | Path | Image.Image) -> InferenceResponse:
        if self._socket is None:
            raise RuntimeError("AsyncClient is not connected. Use an async context manager.")
        payload = await asyncio.to_thread(encode_image, source)
        return await self._exchange(InferenceRequest(payload=payload))

    async def ping(self) -> InferenceResponse:
        return await self._exchange(InferenceRequest(operation="ping"))

    async def __aenter__(self) -> AsyncClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()
