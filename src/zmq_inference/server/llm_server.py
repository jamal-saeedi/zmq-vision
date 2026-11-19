"""Local CPU text generation with a ChatML-compatible GGUF model."""

from pathlib import Path

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.protocol import Generation
from zmq_inference.server.base import BaseModelServer


class LLMInferenceServer(BaseModelServer):
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
            raise ValueError("the llm backend requires CPU and exactly one model worker")
        if type(context_tokens) is not int or not 256 <= context_tokens <= 4096:
            raise ValueError("context_tokens must be an integer from 256 to 4096")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model_path = Path(model_path)
        self.context_tokens = context_tokens
        self.threads = threads
        self.model = None
        self._prefix: list[int] = []
        self._suffix: list[int] = []

    def load_model(self) -> None:
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".gguf":
            raise ValueError(f"expected a local .gguf model file: {self.model_path}")
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
        )
        self._prefix = self.model.tokenize(b"<|im_start|>user\n", add_bos=False, special=True)
        self._suffix = self.model.tokenize(
            b"<|im_end|>\n<|im_start|>assistant\n", add_bos=False, special=True
        )

    def _generate(self, prompt: str, max_tokens: int) -> Generation:
        assert self.model is not None
        content = self.model.tokenize(prompt.encode("utf-8"), add_bos=False, special=False)
        tokens = [*self._prefix, *content, *self._suffix]
        if len(tokens) + max_tokens > self.model.n_ctx():
            raise ValueError("prompt and max_tokens exceed the model context")
        result = self.model.create_completion(
            prompt=tokens,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        choice = result["choices"][0]
        usage = result["usage"]
        return Generation(
            text=choice["text"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            finish_reason=choice["finish_reason"],
        )

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self.model is not None:
                self.model.close()
                self.model = None
