"""A loopback HTTP interface for testing a ZeroMQ server in a browser."""

from __future__ import annotations

import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files

import zmq

from zmq_inference.client import SyncClient
from zmq_inference.core.config import ClientConfig
from zmq_inference.core.protocol import InferenceRequest

MAX_BODY_BYTES = 8 * 1024 * 1024
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class DemoWebServer(HTTPServer):
    """Handle one browser request at a time and stay on IPv4 loopback."""

    def __init__(self, port: int, config: ClientConfig) -> None:
        super().__init__(("127.0.0.1", port), DemoHandler)
        self.config = config

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoWebServer
    timeout = 5

    def log_message(self, _format: str, *args: object) -> None:
        # The browser renders inference errors and outcomes directly.
        pass

    def _local_request(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update(("127.0.0.1", "localhost"))
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host in hosts and (origin is None or origin == f"http://{host}"):
            return True
        self._json(403, {"error": "This demo accepts requests from its local page only."})
        return False

    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(data).encode())

    def do_GET(self) -> None:
        if not self._local_request():
            return
        if self.path in STATIC_ROUTES:
            name, content_type = STATIC_ROUTES[self.path]
            self._send(
                200, content_type, (files("zmq_inference") / "web_assets" / name).read_bytes()
            )
        elif self.path == "/api/health":
            config = replace(
                self.server.config, timeout_ms=min(1200, self.server.config.timeout_ms)
            )
            self._send_request(
                InferenceRequest(operation="ping", metadata={"include_operations": True}), config
            )
        else:
            self._json(404, {"error": "Unknown route."})

    def do_POST(self) -> None:
        if not self._local_request():
            return
        if self.path != "/api/request":
            self._json(404, {"error": "Unknown route."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Send an application/json request."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                self._json(413, {"error": "Request must fit within 8 MiB."})
                return
            request = InferenceRequest.from_json(self.rfile.read(length).decode("utf-8"))
        except (ValueError, TimeoutError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._send_request(request, self.server.config)

    def _send_request(self, request: InferenceRequest, config: ClientConfig) -> None:
        try:
            with SyncClient(config) as client:
                response = client._exchange(request)
        except (OSError, zmq.ZMQError) as exc:
            self._json(503, {"error": str(exc)})
            return
        data = response.to_dict()
        data.pop("image")
        self._json(200 if response.ok else 422, data)
