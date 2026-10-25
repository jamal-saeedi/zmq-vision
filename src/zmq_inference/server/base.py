"""Bounded inference workers; the network thread exclusively owns its sockets."""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import zmq
from PIL import Image

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse, Prediction
from zmq_inference.utils.image import decode_image, encode_image

logger = get_logger("server")


class BaseInferenceServer(ABC, threading.Thread):
    """Extend load_model, preprocess and predict to serve your own image model.

    start() launches the network thread. wait_until_ready() propagates startup
    failures. stop() stops admission; join() waits for running predictions to
    finish. Model code must return for shutdown to complete.
    """

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(daemon=True, name="zmq-vision-server")
        self.config = config
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self.error: Exception | None = None
        self.bound_port: int | None = None

    @abstractmethod
    def load_model(self) -> None:
        """Load model once before the server accepts requests."""

    @abstractmethod
    def preprocess(self, img: Image.Image) -> Any:
        """Convert a decoded RGB image into model input."""

    @abstractmethod
    def predict(self, tensor: Any) -> list[Prediction]:
        """Return predictions in descending confidence order."""

    def stop(self) -> None:
        """Signal shutdown without touching a socket from the calling thread."""
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def wait_until_ready(self, timeout: float = 60) -> None:
        """Wait for model load and bind; raise on startup failure or timeout."""
        if not self._ready.wait(timeout):
            raise TimeoutError("Server did not become ready before the startup deadline")
        if self.error is not None:
            raise RuntimeError(f"Server failed: {self.error}") from self.error
        if self.bound_port is None or not self.is_alive():
            raise RuntimeError("Server stopped before becoming ready")

    def process_request(self, raw: bytes) -> InferenceResponse:
        """Validate and execute one request, returning protocol errors as data."""
        started = time.perf_counter()
        request_id = "unknown"
        try:
            if len(raw) > self.config.max_message_bytes:
                raise ValueError("request exceeds the message size limit")
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("request_id"), str):
                request_id = data["request_id"][:128]
            request = InferenceRequest.from_dict(data)
            request_id = request.request_id
            image_b64 = ""
            predictions = []
            if request.operation == "infer":
                with decode_image(request.payload, max_pixels=self.config.max_image_pixels) as img:
                    predictions = self.predict(self.preprocess(img))
                    image_b64 = encode_image(img)
            return InferenceResponse(
                request_id=request_id,
                predictions=predictions,
                image=image_b64,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            logger.debug("Request %s failed", request_id, exc_info=True)
            return InferenceResponse(
                request_id=request_id,
                predictions=[],
                image="",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

    @staticmethod
    def _send(socket: zmq.Socket, identity: bytes, response: InferenceResponse) -> None:
        try:
            socket.send_multipart([identity, response.to_json().encode()], flags=zmq.DONTWAIT)
        except zmq.ZMQError as exc:
            # A timed-out client can disconnect before inference completes.
            if exc.errno not in (zmq.EAGAIN, zmq.EHOSTUNREACH):
                raise
            logger.debug("Discarding response for disconnected or saturated client")

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        socket.setsockopt(zmq.SNDHWM, self.config.max_pending_requests)
        socket.setsockopt(zmq.RCVHWM, self.config.max_pending_requests)
        # Oversized frames are rejected by libzmq before Python allocates them.
        socket.setsockopt(zmq.MAXMSGSIZE, self.config.max_message_bytes)
        executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="zmq-vision-model"
        )
        pending: dict[Future, bytes] = {}
        try:
            self.load_model()
            if self.stopped():
                return
            if self.config.port == 0:
                self.bound_port = socket.bind_to_random_port(f"tcp://{self.config.host}")
            else:
                socket.bind(self.config.bind_address)
                self.bound_port = self.config.port
            self._ready.set()
            logger.info("Ready on tcp://%s:%s", self.config.host, self.bound_port)
            while not self.stopped():
                for future in list(pending):
                    if future.done():
                        self._send(socket, pending.pop(future), future.result())
                if not socket.poll(self.config.poll_timeout_ms, zmq.POLLIN):
                    continue
                frames = socket.recv_multipart()
                identity = frames[0]
                if len(frames) != 2:
                    self._send(
                        socket,
                        identity,
                        InferenceResponse(
                            "unknown", [], "", 0, "expected one JSON frame after DEALER identity"
                        ),
                    )
                elif len(pending) >= self.config.max_pending_requests:
                    request_id = "unknown"
                    try:
                        data = json.loads(frames[1])
                        if isinstance(data, dict) and isinstance(data.get("request_id"), str):
                            request_id = data["request_id"][:128]
                    except (ValueError, UnicodeDecodeError):
                        pass
                    self._send(
                        socket,
                        identity,
                        InferenceResponse(request_id, [], "", 0, "server busy; retry later"),
                    )
                else:
                    pending[executor.submit(self.process_request, frames[1])] = identity
        except Exception as exc:
            self.error = exc
            logger.exception("Server failed")
        finally:
            self._stop_event.set()
            self._ready.set()
            # Cancel queued jobs, drain active model work, then release transport.
            executor.shutdown(wait=True, cancel_futures=True)
            socket.close(linger=0)
            context.term()
            logger.info("Server stopped")
