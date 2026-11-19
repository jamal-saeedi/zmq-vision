"""Measure warm client round trips against a local TCP server."""

import argparse
import hashlib
import json
import math
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import zmq
from PIL import Image
from PIL import __version__ as pillow_version

from zmq_inference import ClientConfig, ServerConfig, __version__
from zmq_inference.client import SyncClient


def latency_percentiles(samples):
    ordered = sorted(samples)
    return {
        f"p{percentile}_ms": round(ordered[math.ceil(percentile * len(ordered) / 100) - 1], 3)
        for percentile in (50, 95, 99)
    }


def measure(make_request, config, clients, requests, warmup):
    ready = threading.Barrier(clients)

    def worker():
        try:
            with SyncClient(config) as client:
                with Image.new("RGB", (256, 256), (220, 60, 40)) as image:
                    for i in range(warmup):
                        check(make_request(client, image, i))
                    ready.wait(timeout=60)
                    started = time.perf_counter()
                    latencies, processing, tokens = [], [], 0
                    for i in range(requests):
                        before = time.perf_counter()
                        response = make_request(client, image, i + warmup)
                        latencies.append((time.perf_counter() - before) * 1000)
                        check(response)
                        processing.append(response.elapsed_ms)
                        if getattr(response, "generation", None) is not None:
                            tokens += response.generation.completion_tokens
                    return started, time.perf_counter(), latencies, processing, tokens
        except Exception:
            ready.abort()
            raise

    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [pool.submit(worker) for _ in range(clients)]
        return [future.result() for future in futures]


def check(response):
    if not response.ok:
        raise RuntimeError(response.error)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_server(args):
    config = ServerConfig(host="127.0.0.1", port=0, device="cpu", model_name=args.model)

    def infer(client, image, _):
        return client.infer(image)

    if args.backend == "demo":
        from zmq_inference.server.demo_server import DemoInferenceServer

        return DemoInferenceServer(config), infer
    if args.backend == "torch":
        from zmq_inference.server.torch_server import TorchInferenceServer

        return TorchInferenceServer(config), infer
    if args.backend == "tf":
        from zmq_inference.server.tf_server import TFInferenceServer

        return TFInferenceServer(config), infer
    if args.backend == "vlm":
        from zmq_inference.server.vlm_server import VLMInferenceServer

        def describe(client, image, _):
            return client.generate(
                "What is the dominant color in this image? Give one word.",
                image=image,
                max_tokens=args.max_tokens,
            )

        return (
            VLMInferenceServer(
                config,
                model_path=args.model_path,
                projector_path=args.projector_path,
                threads=args.threads,
            ),
            describe,
        )

    from zmq_inference.server.llm_server import LLMInferenceServer

    def generate(client, _, index):
        prompt = f"Example {index}. List three uses of a small computer, in complete sentences."
        return client.generate(prompt, max_tokens=args.max_tokens)

    return LLMInferenceServer(config, model_path=args.model_path, threads=args.threads), generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("demo", "torch", "tf", "llm", "vlm"), default="demo")
    parser.add_argument("--clients", type=int, nargs="+")
    parser.add_argument("--requests", type=int, help="Measured requests per client per round")
    parser.add_argument("--warmup", type=int, help="Warm-up requests per client per round")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--model", default="mobilenetv2_100", help="Pretrained timm model")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--projector-path", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2, help="llama.cpp CPU threads")
    args = parser.parse_args()
    is_vlm = args.backend == "vlm"
    args.clients = args.clients or ([1] if is_vlm else [1, 4])
    if args.requests is None:
        args.requests = 3 if is_vlm else 40
    if args.warmup is None:
        args.warmup = 1 if is_vlm else 5
    if args.rounds is None:
        args.rounds = 2 if is_vlm else 3
    if (
        min(*args.clients, args.requests, args.rounds, args.threads, args.max_tokens) < 1
        or args.warmup < 0
    ):
        parser.error(
            "clients, requests, rounds, threads and tokens must be positive; warmup can be zero"
        )
    if args.backend in {"llm", "vlm"} and args.model_path is None:
        parser.error(f"--model-path is required for the {args.backend} backend")
    if is_vlm and args.projector_path is None:
        parser.error("--projector-path is required for the vlm backend")

    server, make_request = make_server(args)
    before_start = time.perf_counter()
    server.start()
    try:
        server.wait_until_ready(timeout=120)
        result = {
            "package": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyzmq": zmq.pyzmq_version(),
            "libzmq": zmq.zmq_version(),
            "pillow": pillow_version,
            "backend": args.backend,
            "startup_ms": round((time.perf_counter() - before_start) * 1000, 2),
            "rounds": args.rounds,
            "warmup_per_client": args.warmup,
            "workloads": [],
        }
        if args.backend in {"llm", "vlm"}:
            result["model_sha256"] = sha256(args.model_path)
            if is_vlm:
                result["projector_sha256"] = sha256(args.projector_path)
            result.update(
                model=args.model_path.name, max_tokens=args.max_tokens, cpu_threads=args.threads
            )
            result["runtime"] = f"llama-cpp-python {version('llama-cpp-python')}"
        elif args.backend == "torch":
            result["model"] = args.model
            result["runtime"] = f"torch {version('torch')} / timm {version('timm')}"
        elif args.backend == "tf":
            result["model"] = "Keras MobileNet"
            result["runtime"] = f"TensorFlow {version('tensorflow')}"
        config = ClientConfig(host="127.0.0.1", port=server.bound_port)
        for count in args.clients:
            studies = [
                measure(make_request, config, count, args.requests, args.warmup)
                for _ in range(args.rounds)
            ]
            rows = [row for study in studies for row in study]
            latencies = [sample for _, _, samples, _, _ in rows for sample in samples]
            processing = [sample for _, _, _, samples, _ in rows for sample in samples]
            wall = sum(
                max(row[1] for row in study) - min(row[0] for row in study) for study in studies
            )
            workload = {
                "clients": count,
                "requests": len(latencies),
                "requests_per_second": round(len(latencies) / wall, 2),
                "round_trip": latency_percentiles(latencies),
                "server_processing": latency_percentiles(processing),
            }
            if args.backend in {"llm", "vlm"}:
                tokens = sum(row[-1] for row in rows)
                workload["completion_tokens"] = tokens
                workload["completion_tokens_per_second"] = round(tokens / wall, 2)
            result["workloads"].append(workload)
        print(json.dumps(result, indent=2))
    finally:
        server.stop()
        server.join()


if __name__ == "__main__":
    main()
