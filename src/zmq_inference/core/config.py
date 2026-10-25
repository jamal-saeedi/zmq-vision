"""Validated configuration. Explicit arguments override environment defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class ServerConfig:
    """Server limits. Port zero requests an available TCP port.

    One model worker is the default; increase only for thread-safe backends.
    Environment variables are read at construction; .env files are not loaded.
    """

    host: str = field(default_factory=lambda: os.getenv("ZMQ_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("ZMQ_PORT", "5576")))
    model_name: str = field(default_factory=lambda: os.getenv("ZMQ_MODEL", "mobilenetv2_100"))
    device: str = field(default_factory=lambda: os.getenv("ZMQ_DEVICE", "auto"))
    top_k: int = 3
    poll_timeout_ms: int = 50
    max_workers: int = 1
    max_pending_requests: int = 16
    max_message_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    pretrained: bool = True

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 0 <= self.port <= 65535
        ):
            raise ValueError("port must be between 0 and 65535")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        for name in (
            "top_k",
            "poll_timeout_ms",
            "max_workers",
            "max_pending_requests",
            "max_message_bytes",
            "max_image_pixels",
        ):
            _positive(name, getattr(self, name))
        if self.top_k > 1000:
            raise ValueError("top_k must not exceed 1000 for ImageNet backends")

    @property
    def bind_address(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass
class ClientConfig:
    """Each attempt has a bounded timeout; retries may repeat model execution."""

    host: str = field(default_factory=lambda: os.getenv("ZMQ_SERVER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("ZMQ_PORT", "5576")))
    timeout_ms: int = field(default_factory=lambda: int(os.getenv("ZMQ_TIMEOUT_MS", "30000")))
    max_retries: int = 1

    def __post_init__(self) -> None:
        if not self.host or self.host == "*":
            raise ValueError("client host must name a reachable host")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be between 1 and 65535")
        _positive("timeout_ms", self.timeout_ms)
        _positive("max_retries", self.max_retries)

    @property
    def server_address(self) -> str:
        return f"tcp://{self.host}:{self.port}"
