"""Synchronous client with deadlines, correlation, and fresh sockets on timeout."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import zmq
from PIL import Image

from zmq_inference.client.base import BaseInferenceClient
from zmq_inference.core.config import ClientConfig
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse
from zmq_inference.utils.image import encode_image


class SyncClient(BaseInferenceClient):
    """Use from one thread; create a client per thread for concurrent callers.

    max_retries counts total attempts. A retry may execute inference again.
    A fresh socket after timeout isolates delayed replies from later requests.
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        super().__init__(config)
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._owner: int | None = None

    def _open_socket(self) -> None:
        assert self._context is not None
        socket = self._context.socket(zmq.DEALER)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt_string(zmq.IDENTITY, str(uuid.uuid4()))
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.SNDTIMEO, self.config.timeout_ms)
            socket.connect(self.config.server_address)
        except Exception:
            socket.close(linger=0)
            raise
        self._socket = socket

    def connect(self) -> None:
        if self._socket is not None:
            self._check_owner()
            return
        self._context = zmq.Context()
        self._owner = threading.get_ident()
        try:
            self._open_socket()
        except Exception:
            self.disconnect()
            raise

    def _check_owner(self) -> None:
        if self._owner is not None and self._owner != threading.get_ident():
            raise RuntimeError("SyncClient must be used and closed in its connecting thread")

    def disconnect(self) -> None:
        self._check_owner()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._owner = None

    def _exchange(self, request: InferenceRequest) -> InferenceResponse:
        self._check_owner()
        if self._socket is None:
            raise RuntimeError(
                "SyncClient is not connected. Call connect() or use a context manager."
            )
        for _ in range(self.config.max_retries):
            deadline = time.monotonic() + self.config.timeout_ms / 1000
            try:
                self._socket.setsockopt(zmq.SNDTIMEO, self.config.timeout_ms)
                self._socket.send_json(request.to_dict())
                while (remaining := deadline - time.monotonic()) > 0:
                    if not self._socket.poll(max(1, int(remaining * 1000)), zmq.POLLIN):
                        break
                    response = InferenceResponse.from_json(self._socket.recv_string())
                    if response.request_id == request.request_id:
                        return response
            except zmq.Again:
                pass
            self._socket.close(linger=0)
            self._open_socket()
        raise TimeoutError(
            f"Server at {self.config.server_address} did not respond after "
            f"{self.config.max_retries} attempt(s) ({self.config.timeout_ms} ms each)."
        )

    def infer(self, source: str | Path | Image.Image) -> InferenceResponse:
        self._check_owner()
        if self._socket is None:
            raise RuntimeError(
                "SyncClient is not connected. Call connect() or use a context manager."
            )
        return self._exchange(InferenceRequest(payload=encode_image(source)))

    def ping(self) -> InferenceResponse:
        """Check readiness without running model inference."""
        return self._exchange(InferenceRequest(operation="ping"))
