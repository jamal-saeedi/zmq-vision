"""CPU runtime configuration and lifecycle shared by the GGUF backends."""

from pathlib import Path

from zmq_inference.core.config import ServerConfig
from zmq_inference.server.base import BaseModelServer


def require_gguf(path: Path) -> None:
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError(f"expected a local .gguf file: {path}")


class GGUFInferenceServer(BaseModelServer):
    """Keep the mutable llama.cpp context on a single inference worker."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        model_path: str | Path,
        context_tokens: int = 2048,
        threads: int = 2,
    ) -> None:
        super().__init__(config)
        if config.device not in {"auto", "cpu"} or config.max_workers != 1:
            raise ValueError("GGUF backends require CPU and exactly one model worker")
        if type(context_tokens) is not int or not 256 <= context_tokens <= 4096:
            raise ValueError("context_tokens must be an integer from 256 to 4096")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model_path = Path(model_path)
        self.context_tokens = context_tokens
        self.threads = threads
        self.model = None

    def _load_gguf(self, **options) -> None:
        require_gguf(self.model_path)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                'install the optional CPU runtime: pip install "zmq-vision[llm]"'
            ) from exc

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_tokens,
            n_batch=min(self.context_tokens, 128),
            n_threads=self.threads,
            n_threads_batch=self.threads,
            n_gpu_layers=0,
            verbose=False,
            **options,
        )

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self.model is not None:
                self.model.close()
                self.model = None
