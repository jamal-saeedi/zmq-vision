"""Image-conditioned generation with SmolVLM and the llama.cpp MTMD runtime."""

from ctypes import CFUNCTYPE, c_char_p, c_int, c_void_p
from pathlib import Path

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import Generation
from zmq_inference.server.gguf_server import GGUFInferenceServer, require_gguf
from zmq_inference.utils.image import decode_image, encode_image

logger = get_logger("vlm")


@CFUNCTYPE(None, c_int, c_char_p, c_void_p)
def _log_mtmd_error(level: int, message: bytes | None, _user_data) -> None:
    # Native informational logs can contain the full prompt.
    if level == 3 and message:
        logger.error("%s", message.decode("utf-8", errors="replace").rstrip())


class VLMInferenceServer(GGUFInferenceServer):
    """Serve one text prompt with an optional image using local GGUF files."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        model_path: str | Path,
        projector_path: str | Path,
        context_tokens: int = 2048,
        threads: int = 2,
    ) -> None:
        super().__init__(
            config, model_path=model_path, context_tokens=context_tokens, threads=threads
        )
        self.projector_path = Path(projector_path)

    def load_model(self) -> None:
        require_gguf(self.projector_path)
        try:
            from llama_cpp import mtmd_cpp
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except ImportError as exc:
            raise ImportError(
                'install the MTMD CPU runtime: pip install "zmq-vision[vlm]"'
            ) from exc
        handler = MTMDChatHandler(
            clip_model_path=str(self.projector_path), verbose=False, use_gpu=False
        )
        self._load_gguf(chat_handler=handler)
        mtmd_cpp.mtmd_log_set(_log_mtmd_error, None)
        mtmd_cpp.mtmd_helper_log_set(_log_mtmd_error, None)
        # The first call initializes the projector even when there is no image.
        self._complete("Hello.", 1)

    def _generate(self, prompt: str, max_tokens: int) -> Generation:
        return self._complete(prompt, max_tokens)

    def _describe(self, payload: str, prompt: str, max_tokens: int) -> Generation:
        with decode_image(payload, max_pixels=self.config.max_image_pixels) as image:
            image.thumbnail((512, 512))
            return self._complete(prompt, max_tokens, encode_image(image))

    def _complete(self, prompt: str, max_tokens: int, image_b64: str | None = None) -> Generation:
        assert self.model is not None
        content = []
        if image_b64 is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})
        result = self.model.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        choice = result["choices"][0]
        usage = result["usage"]
        return Generation(
            text=choice["message"]["content"] or "",
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            finish_reason=choice["finish_reason"],
        )
