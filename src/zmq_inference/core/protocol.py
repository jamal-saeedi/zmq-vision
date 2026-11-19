"""Request/Response dataclasses and JSON serialization for the ZMQ protocol."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_PROMPT_BYTES = 8192
DEFAULT_MAX_TOKENS = 64
MAX_GENERATION_TOKENS = 256


@dataclass
class InferenceRequest:
    """Image inference, text or image-conditioned generation, or readiness."""

    payload: str = ""  # base64 encoded image; empty for ping
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    operation: str = "infer"
    prompt: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if not isinstance(self.payload, str):
            raise ValueError("payload must be a base64 string")
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or len(self.request_id) > 128
        ):
            raise ValueError("request_id must be a nonempty string of at most 128 characters")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if self.operation not in ("infer", "ping", "generate", "describe"):
            raise ValueError("operation must be infer, ping, generate, or describe")
        if self.operation == "infer" and not self.payload:
            raise ValueError("inference requires a nonempty payload")
        if self.operation == "describe" and not self.payload:
            raise ValueError("image-conditioned generation requires a nonempty payload")
        if self.operation in ("generate", "describe"):
            if self.operation == "generate" and self.payload:
                raise ValueError("generation takes a text prompt and an empty image payload")
            if not isinstance(self.prompt, str) or not self.prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            if len(self.prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
                raise ValueError(f"prompt exceeds {MAX_PROMPT_BYTES} UTF-8 bytes")
            if (
                type(self.max_tokens) is not int
                or not 1 <= self.max_tokens <= MAX_GENERATION_TOKENS
            ):
                raise ValueError(f"max_tokens must be an integer from 1 to {MAX_GENERATION_TOKENS}")

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        data = {
            "payload": self.payload,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }
        if self.operation != "infer":
            data["operation"] = self.operation
        if self.operation in ("generate", "describe"):
            data.update(prompt=self.prompt, max_tokens=self.max_tokens)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceRequest:
        """Deserialise from a plain dict."""
        if not isinstance(data, dict):
            raise ValueError("request must be a JSON object")
        return cls(
            payload=data.get("payload", ""),
            request_id=data.get("request_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
            operation=data.get("operation", "infer"),
            prompt=data.get("prompt"),
            max_tokens=data.get("max_tokens", DEFAULT_MAX_TOKENS),
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> InferenceRequest:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(raw))


@dataclass
class Prediction:
    """A single top-k class prediction.

    Attributes:
        label:      Human-readable class label (e.g. ``"golden retriever"``).
        confidence: Model confidence in the range [0, 1].
    """

    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        return cls(label=data["label"], confidence=float(data["confidence"]))

    def __str__(self) -> str:
        return f"{self.label}: {self.confidence:.5f}"


@dataclass
class Generation:
    """Generated text and usage reported by the model runtime."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass
class InferenceResponse:
    """A server response carrying predictions or generated text.

    Attributes:
        request_id:  Echoed from the originating :class:`InferenceRequest`.
        predictions: Ordered list of top-k :class:`Prediction` objects.
        image:       Base64-encoded result/echo image.
        elapsed_ms:  Server-side processing time in milliseconds.
        error:       Non-``None`` when the server encountered an error.
        generation:  Text and token usage on a successful generation request.
        operations:  Supported model operations when explicitly requested via ping.
    """

    request_id: str
    predictions: list[Prediction]
    image: str  # base64 encoded result image
    elapsed_ms: float
    error: str | None = None
    generation: Generation | None = None
    operations: list[str] | None = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = {
            "request_id": self.request_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "image": self.image,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }
        if self.generation is not None:
            data["generation"] = {
                "text": self.generation.text,
                "prompt_tokens": self.generation.prompt_tokens,
                "completion_tokens": self.generation.completion_tokens,
                "finish_reason": self.generation.finish_reason,
            }
        if self.operations is not None:
            data["operations"] = self.operations
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceResponse:
        generation = data.get("generation")
        return cls(
            request_id=data["request_id"],
            predictions=[Prediction.from_dict(p) for p in data.get("predictions", [])],
            image=data.get("image", ""),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            error=data.get("error"),
            generation=Generation(**generation) if generation is not None else None,
            operations=data.get("operations"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> InferenceResponse:
        return cls.from_dict(json.loads(raw))

    @property
    def ok(self) -> bool:
        """``True`` when no error was reported."""
        return self.error is None

    def __str__(self) -> str:
        if self.error:
            return f"InferenceResponse(error={self.error!r})"
        if self.generation is not None:
            return f"InferenceResponse(generation={self.generation!r}, elapsed_ms={self.elapsed_ms:.1f})"
        preds = ", ".join(str(p) for p in self.predictions)
        return f"InferenceResponse(predictions=[{preds}], elapsed_ms={self.elapsed_ms:.1f})"
