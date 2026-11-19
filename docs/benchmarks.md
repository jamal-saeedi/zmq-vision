# Local TCP benchmarks

Measured on November 19, 2026 on WSL2, Intel Core i7-10700K, CPython 3.12.3, PyZMQ 27.3.0, libzmq 4.3.5, and Pillow 13.0.0. Each run used one model worker. PyTorch and llama.cpp inference used two CPU threads. The baseline is commit `a9c66fe`, which reports version 0.3.0.

## Vision

Clients repeatedly classified an in-memory 256 × 256 RGB image. This times client image encoding, the TCP round trip, queueing, model work, and response parsing. The RGB demo isolates the transport with a cheap workload. MobileNetV2 uses pretrained ImageNet weights with torch 2.15.0 and timm 1.0.31.

| Workload | Clients | Version | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Req/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| RGB demo | 1 | 0.3.0 | 120 | 55.95 | 56.50 | 56.63 | 18.23 |
| RGB demo | 1 | 0.4.0 | 120 | 3.17 | 3.31 | 3.40 | 317.51 |
| RGB demo | 4 | 0.3.0 | 480 | 52.95 | 58.95 | 59.94 | 117.85 |
| RGB demo | 4 | 0.4.0 | 480 | 3.90 | 5.34 | 6.12 | 964.48 |
| MobileNetV2 | 1 | 0.3.0 | 40 | 56.11 | 56.43 | 56.56 | 17.91 |
| MobileNetV2 | 1 | 0.4.0 | 40 | 19.27 | 21.75 | 21.93 | 52.13 |
| MobileNetV2 | 4 | 0.3.0 | 160 | 56.56 | 111.05 | 113.77 | 54.34 |
| MobileNetV2 | 4 | 0.4.0 | 160 | 70.53 | 78.03 | 80.51 | 56.19 |

The new server polls for completed work sooner. The RGB demo processed each single-client request in about 0.6 ms on both versions. Four closed-loop MobileNet clients saturate one model worker. Their median round trip increased by 14.0 ms while p95 fell by 33.0 ms in this run. The 2 ms pending-work poll can raise network-thread CPU use. CPU overhead was not measured.

## Local text generation

SmolLM2-135M-Instruct Q4_K_M, llama-cpp-python 0.3.37, 2,048-token context, 32 output tokens per request. The harness varied the example number in a fixed prompt template. Every measured request reached the token cap.

| Clients | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Req/s | Output tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 318.05 | 365.97 | 389.18 | 3.07 | 98.13 |
| 4 | 96 | 1,123.33 | 1,217.00 | 1,259.51 | 3.51 | 112.41 |

Token throughput is aggregate returned completion tokens divided by elapsed test time. It includes prompt processing, queueing, and TCP. The server became ready in 181 ms using the downloaded model. Weights, native runtime build flags, CPU scheduling, and prompt reuse can affect results.

## Reproduce

Install an editable checkout and the extra for each workload you plan to run. Download the pinned GGUF from the [README](../README.md#generate-text-on-cpu).

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python scripts/benchmark.py --backend demo
python scripts/benchmark.py --backend torch --requests 20 --rounds 2
python scripts/benchmark.py --backend llm --model-path smollm2.gguf --requests 8 --warmup 2
```

Run the same harness with `PYTHONPATH` pointing to the older checkout's `src` directory for a baseline. Use the same Python environment for both versions. Each client keeps one request in flight and warms up before timing begins. Percentiles use the nearest-rank method over completed requests from all rounds. Error responses abort the run. These are results from one workstation with up to four clients. [Raw JSON results](benchmark-results.json)
