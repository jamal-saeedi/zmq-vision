"""Command-line interface shared by zmq-vision and the zmq-inference alias."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import platform
import signal
from pathlib import Path

import click
import zmq

from zmq_inference import __version__
from zmq_inference.core.config import ClientConfig, ServerConfig
from zmq_inference.core.logging import setup_logging
from zmq_inference.core.protocol import DEFAULT_MAX_TOKENS, MAX_GENERATION_TOKENS, InferenceResponse


class InferenceGroup(click.Group):
    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except (ValueError, RuntimeError, OSError, ImportError, zmq.ZMQError) as exc:
            raise click.ClickException(str(exc)) from exc


@click.group(cls=InferenceGroup)
@click.version_option(__version__)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def cli(log_level: str) -> None:
    """Serve image classifiers, LLMs, and vision-language models over ZeroMQ."""
    setup_logging(log_level)


def _config(cls, **values):
    return cls(**{key: value for key, value in values.items() if value is not None})


@cli.command("serve")
@click.option(
    "--backend",
    default="demo",
    show_default=True,
    type=click.Choice(["demo", "torch", "tf", "llm", "vlm"]),
)
@click.option("--host", help="Bind host; defaults to ZMQ_HOST or 127.0.0.1.")
@click.option("--port", type=click.IntRange(0, 65535), help="Bind port; zero selects a free port.")
@click.option("--model", help="1000-class timm model (torch only).")
@click.option(
    "--model-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Local GGUF weights for the llm or vlm backend.",
)
@click.option(
    "--projector-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Matching local vision projector GGUF (vlm only).",
)
@click.option(
    "--context-tokens", type=click.IntRange(256, 4096), help="GGUF context size (default: 2048)."
)
@click.option("--threads", type=click.IntRange(1), help="GGUF CPU threads (default: 2).")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]))
@click.option("--top-k", default=3, type=click.IntRange(1, 1000), show_default=True)
@click.option(
    "--workers",
    default=1,
    type=click.IntRange(1),
    show_default=True,
    help="Model threads. Increase only for thread-safe backends.",
)
@click.option("--max-pending", default=16, type=click.IntRange(1), show_default=True)
@click.option("--startup-timeout", default=300.0, type=click.FloatRange(min=0.1), show_default=True)
@click.option(
    "--no-pretrained", is_flag=True, help="Random model weights for offline smoke tests only."
)
def serve_cmd(
    backend,
    host,
    port,
    model,
    model_path,
    projector_path,
    context_tokens,
    threads,
    device,
    top_k,
    workers,
    max_pending,
    startup_timeout,
    no_pretrained,
):
    """Run one backend. Torch, TensorFlow, LLM and VLM require separate extras."""
    from zmq_inference.server.demo_server import DemoInferenceServer
    from zmq_inference.server.llm_server import LLMInferenceServer
    from zmq_inference.server.tf_server import TFInferenceServer
    from zmq_inference.server.torch_server import TorchInferenceServer
    from zmq_inference.server.vlm_server import VLMInferenceServer

    config = _config(
        ServerConfig,
        host=host,
        port=port,
        model_name=model,
        device=device,
        top_k=top_k,
        max_workers=workers,
        max_pending_requests=max_pending,
        pretrained=not no_pretrained,
    )
    if projector_path is not None and backend != "vlm":
        raise click.UsageError("--projector-path requires --backend vlm")
    if backend in {"llm", "vlm"}:
        if model_path is None:
            raise click.UsageError(f"--backend {backend} requires --model-path")
        if no_pretrained:
            raise click.UsageError(f"--backend {backend} requires GGUF weights")
        if backend == "vlm" and projector_path is None:
            raise click.UsageError("--backend vlm requires --projector-path")
        server = _config(
            VLMInferenceServer if backend == "vlm" else LLMInferenceServer,
            config=config,
            model_path=model_path,
            projector_path=projector_path,
            context_tokens=context_tokens,
            threads=threads,
        )
    else:
        if model_path is not None or context_tokens is not None or threads is not None:
            raise click.UsageError(
                "--model-path, --context-tokens and --threads require --backend llm or vlm"
            )
        # ML modules defer their optional dependency imports until load_model().
        server = {
            "demo": DemoInferenceServer,
            "torch": TorchInferenceServer,
            "tf": TFInferenceServer,
        }[backend](config)
    previous = {}

    def shutdown(_signal, _frame):
        server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, shutdown)
    server.start()
    try:
        server.wait_until_ready(startup_timeout)
        click.echo(f"ZMQ Inference | backend={backend} | tcp://{config.host}:{server.bound_port}")
        if no_pretrained and backend != "demo":
            click.echo("Random weights: predictions are not meaningful.", err=True)
        while server.is_alive():
            server.join(timeout=0.2)
        if server.error is not None:
            raise RuntimeError(str(server.error))
    finally:
        server.stop()
        server.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@cli.command("infer")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--host", help="Server hostname.")
@click.option("--port", type=click.IntRange(1, 65535))
@click.option("--timeout", type=click.IntRange(1), help="Per-attempt timeout in milliseconds.")
@click.option(
    "--retries",
    default=1,
    type=click.IntRange(1),
    show_default=True,
    help="Total attempts; retries may repeat inference.",
)
@click.option(
    "--top-k",
    default=3,
    type=click.IntRange(1, 1000),
    show_default=True,
    help="Maximum returned predictions to display; server controls prediction count.",
)
@click.option("--async", "use_async", is_flag=True, help="Use the asyncio client.")
@click.option("--json", "as_json", is_flag=True, help="Emit the protocol response as JSON.")
def infer_cmd(image_path, host, port, timeout, retries, top_k, use_async, as_json):
    """Send IMAGE_PATH to the server and print its predictions."""
    config = _config(ClientConfig, host=host, port=port, timeout_ms=timeout, max_retries=retries)
    if use_async:
        from zmq_inference.client.async_client import AsyncClient

        async def run():
            async with AsyncClient(config) as client:
                return await client.infer(image_path)

        response = asyncio.run(run())
    else:
        from zmq_inference.client.sync_client import SyncClient

        with SyncClient(config) as client:
            response = client.infer(image_path)
    if not response.ok:
        raise click.ClickException(f"Server error: {response.error}")
    if as_json:
        click.echo(response.to_json())
    else:
        _print_response(response, top_k)


def _print_response(response: InferenceResponse, top_k: int) -> None:
    click.echo(f"Request ID : {response.request_id}")
    click.echo(f"Elapsed    : {response.elapsed_ms:.1f} ms (server processing)")
    for index, prediction in enumerate(response.predictions[:top_k], 1):
        click.echo(f"  {index}. {prediction.label:40s} {prediction.confidence:.5f}")


@cli.command("generate")
@click.argument("prompt")
@click.option(
    "--image",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Condition the response on a local image (requires the vlm backend).",
)
@click.option("--host", help="Server hostname.")
@click.option("--port", type=click.IntRange(1, 65535))
@click.option("--timeout", type=click.IntRange(1), help="Per-attempt timeout in milliseconds.")
@click.option("--retries", default=1, type=click.IntRange(1), show_default=True)
@click.option(
    "--max-tokens",
    default=DEFAULT_MAX_TOKENS,
    type=click.IntRange(1, MAX_GENERATION_TOKENS),
    show_default=True,
)
@click.option("--async", "use_async", is_flag=True, help="Use the asyncio client.")
@click.option("--json", "as_json", is_flag=True, help="Include request ID and token usage.")
def generate_cmd(prompt, image, host, port, timeout, retries, max_tokens, use_async, as_json):
    """Send a prompt with an optional image."""
    config = _config(ClientConfig, host=host, port=port, timeout_ms=timeout, max_retries=retries)
    if use_async:
        from zmq_inference.client.async_client import AsyncClient

        async def run():
            async with AsyncClient(config) as client:
                return await client.generate(prompt, image=image, max_tokens=max_tokens)

        response = asyncio.run(run())
    else:
        from zmq_inference.client.sync_client import SyncClient

        with SyncClient(config) as client:
            response = client.generate(prompt, image=image, max_tokens=max_tokens)
    if not response.ok:
        raise click.ClickException(f"Server error: {response.error}")
    if response.generation is None:
        raise click.ClickException("Server did not return a generation")
    click.echo(response.to_json() if as_json else response.generation.text)


@cli.command("health")
@click.option("--host")
@click.option("--port", type=click.IntRange(1, 65535))
@click.option("--timeout", default=3000, type=click.IntRange(1), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def health_cmd(host, port, timeout, as_json):
    """Check readiness without running a model prediction."""
    from zmq_inference.client.sync_client import SyncClient

    config = _config(ClientConfig, host=host, port=port, timeout_ms=timeout)
    with SyncClient(config) as client:
        response = client.ping()
    if not response.ok:
        raise click.ClickException(f"Server error: {response.error}")
    click.echo(response.to_json() if as_json else f"OK — {config.server_address} is ready")


@cli.command("web")
@click.option(
    "--demo", is_flag=True, help="Start a download-free classifier as part of the browser demo."
)
@click.option("--port", default=8080, type=click.IntRange(0, 65535), show_default=True)
@click.option(
    "--zmq-port", type=click.IntRange(1, 65535), help="Connect to an existing local server."
)
@click.option("--timeout", type=click.IntRange(1), help="ZeroMQ timeout in milliseconds.")
def web_cmd(demo, port, zmq_port, timeout):
    """Serve a local browser interface for inference over ZeroMQ."""
    from zmq_inference.server.demo_server import DemoInferenceServer
    from zmq_inference.web import DemoWebServer

    if demo and zmq_port is not None:
        raise click.UsageError("use --demo or --zmq-port")
    server = None
    try:
        if demo:
            server = DemoInferenceServer(ServerConfig(host="127.0.0.1", port=0))
            server.start()
            server.wait_until_ready(10)
            zmq_port = server.bound_port
        config = _config(ClientConfig, host="127.0.0.1", port=zmq_port, timeout_ms=timeout)
        with DemoWebServer(port, config) as web:
            click.echo(f"Open {web.url} to test {config.server_address}")
            web.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
            server.join()


@cli.command("demo")
@click.option("--json", "as_json", is_flag=True)
def demo_cmd(as_json):
    """Run a self-contained demo: real TCP, sync + async, generated images."""
    from zmq_inference.demo import run_demo

    result = run_demo()
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("ZMQ Inference · local TCP demo")
        click.echo("Backend: RGB statistics (no trained model, no downloads)")
        click.echo("Health: OK")
        for mode in ("sync", "async"):
            response = InferenceResponse.from_dict(result[mode])
            click.echo(
                f"{mode:5s}: {response.predictions[0]} | {response.elapsed_ms:.2f} ms server time"
            )
        click.echo("PASS · correlated responses received; server shut down cleanly")


@cli.command("doctor")
def doctor_cmd():
    """Print interpreter, libzmq and installed optional backend versions."""
    versions = {
        "python": platform.python_version(),
        "zmq-vision": __version__,
        "pyzmq": zmq.pyzmq_version(),
        "libzmq": zmq.zmq_version(),
    }
    for package in (
        "Pillow",
        "click",
        "torch",
        "torchvision",
        "timm",
        "tensorflow",
        "llama-cpp-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    click.echo(json.dumps(versions, indent=2))
