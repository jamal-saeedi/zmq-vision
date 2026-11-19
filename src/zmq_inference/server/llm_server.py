"""Local CPU text generation with a ChatML-compatible GGUF model."""

from zmq_inference.core.protocol import Generation
from zmq_inference.server.gguf_server import GGUFInferenceServer


class LLMInferenceServer(GGUFInferenceServer):
    """Serve a text-only ChatML model."""

    def load_model(self) -> None:
        self._load_gguf()
        assert self.model is not None
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
