"""ZMQ Inference serves local vision and language models over ZeroMQ."""

__version__ = "0.5.0"
__author__ = "Jamal Saeedi"
__email__ = "jamal.saeedi7@gmail.com"

from zmq_inference.core.config import ClientConfig, ServerConfig
from zmq_inference.core.protocol import Generation, InferenceRequest, InferenceResponse, Prediction

__all__ = [
    "__version__",
    "ServerConfig",
    "ClientConfig",
    "InferenceRequest",
    "InferenceResponse",
    "Prediction",
    "Generation",
]
