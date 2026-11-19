# Protocol and execution model

Each client owns a DEALER socket and sends one UTF-8 JSON frame. The server's ROUTER receives the connection identity followed by that frame. The network thread owns the ROUTER. A bounded `ThreadPoolExecutor` decodes images or tokenizes prompts and executes the model. The network thread sends completed futures back to the originating client.

## Messages

Inference request:

```json
{"request_id":"example-1","payload":"<base64 image>","metadata":{}}
```

Readiness request:

```json
{"request_id":"example-2","operation":"ping","payload":"","metadata":{}}
```

Response (image and predictions are empty on a readiness check or failure):

```json
{"request_id":"example-1","predictions":[{"label":"red","confidence":0.82}],"image":"<base64 JPEG>","elapsed_ms":0.7,"error":null}
```

Text generation on the LLM backend:

```json
{"request_id":"example-3","operation":"generate","prompt":"Name a primary color.","max_tokens":24}
```

```json
{"request_id":"example-3","predictions":[],"image":"","elapsed_ms":41,"error":null,"generation":{"text":"Red.","prompt_tokens":13,"completion_tokens":2,"finish_reason":"stop"}}
```

`generation` is present only on a successful text response. The image and ping response schemas from 0.3 are unchanged. Requests are JSON objects with a nonempty string ID (maximum 128 characters) and optional object metadata. Omitting an ID generates a new one for older callers. The image payload is strict base64 and decodes to a Pillow-readable image. Generation accepts an optional empty image payload and a nonblank Unicode prompt. It limits the prompt to 8,192 UTF-8 bytes and `max_tokens` to 1–256, with a default of 64. The backend rejects requests whose formatted prompt plus output budget exceeds its context window.

`elapsed_ms` measures server work including image serialization or text generation. It excludes queueing and network round trips. Token usage comes from the runtime. The text finish reason is `stop` for a model stop token or stop sequence and `length` when the token budget is reached.

Malformed JSON or image data produces an error response without terminating the server. Invalid multipart envelopes receive an error with ID `unknown`. Invalid IDs and oversized frames may be impossible to correlate; clients eventually time out. A custom caller should always send and check an ID.

## Limits and lifecycle

`ServerConfig` defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_workers` | 1 | Concurrent model calls |
| `max_pending_requests` | 16 | Accepted running plus queued futures |
| `max_message_bytes` | 8 MiB | libzmq frame limit and application size check |
| `max_image_pixels` | 16,000,000 | Decoded image width × height |
| `poll_timeout_ms` | 50 | Idle socket poll timeout. While work is pending, the poll timeout is at most 2 ms. |

Workers can complete while the network thread is polling the socket. The shorter poll while work is pending reduces response delivery delay at the cost of more frequent polls during inference. Idle servers retain the idle poll interval.

libzmq high water marks also bound per-peer transport queues. Total memory still depends on the number of connected peers and decoded images; these are not global process memory quotas. Image echoing is retained for API compatibility and adds bandwidth and encoding work.

`start()` begins model loading in the server thread. `wait_until_ready()` waits until model loading and TCP binding succeed, and propagates a failure. Model startup may download weights; cache those before deploying offline. A stop during load takes effect after loading returns.

`stop()` signals from any thread without touching a socket. The network loop stops admission, cancels queued futures, and waits for running model calls before closing its socket and context. Running requests may not receive a response during shutdown. A model call that never returns prevents complete shutdown; use process supervision if hard termination is required.

A startup timeout limits waiting for readiness, not the model-loading operation itself. The CLI cleans up and waits for that thread; a stuck native model loader can therefore outlive the startup deadline.

## Client semantics

On Windows, the platform-specific Tornado dependency supplies the selector bridge required by PyZMQ on the default Proactor event loop. Other platforms do not need that dependency.

A synchronous client can only be used and closed in its connecting thread. Async clients belong to one event loop and serialize exchanges under a lock. Use separate clients for parallel work. Connection and cleanup operations are idempotent.

Every attempt has a network deadline, including send time. The default is one attempt. `ClientConfig.max_retries` is the total number of attempts, retained from the older API. On timeout or async cancellation, the old socket closes with zero linger and the next attempt gets a fresh routing identity. Clients also discard responses for other request IDs.

Retries provide **at-least-once attempts**, not exactly-once execution. The server does not cache request IDs. Side effects in a custom predictor can therefore repeat. Encoding a local input image is outside the network timeout.

## Models and extension boundaries

PyTorch uses `timm.data.resolve_model_data_config` and `create_transform` so input size, crop, interpolation, and normalization follow the selected model. The built-in label mapping is ImageNet-1k; other classifier heads are rejected. TensorFlow uses MobileNet and `training=False`, avoiding per-request dataset construction and extra label-file downloads.

One worker avoids concurrent access to a model by default. More workers share the model instance; the backend author must establish thread safety and memory requirements before increasing the count. This is not dynamic batching or a multi-process GPU scheduler.

The LLM backend loads a local ChatML-compatible GGUF into one CPU context and defaults to two inference threads and a 2,048-token context. The server requires one model worker because llama.cpp keeps mutable state. The backend formats a single user message and returns one complete response. A client timeout or cancellation resets the client socket. The server may still finish generation.

## Configuration

Explicit Python arguments and CLI options override environment defaults. The application reads exported variables. See the [.env example](../.env.example) for shell setup.

| Variable | Default | Used by |
| --- | --- | --- |
| `ZMQ_HOST` | `127.0.0.1` | Server bind host |
| `ZMQ_SERVER_HOST` | `127.0.0.1` | Client destination |
| `ZMQ_PORT` | `5576` | Server and client |
| `ZMQ_MODEL` | `mobilenetv2_100` | PyTorch model |
| `ZMQ_DEVICE` | `auto` | Model device selection |
| `ZMQ_TIMEOUT_MS` | `30000` | Client deadline per attempt |

The protocol sends plaintext over TCP. Bind to localhost or use a protected private network or tunnel.

## Upstream references

- [ZeroMQ thread-safety rules](https://zeromq.github.io/libzmq/zmq.html): contexts may be shared; ordinary sockets require thread ownership.
- [PyZMQ asyncio on Windows](https://github.com/zeromq/pyzmq/issues/1830): the selector bridge supports the default Proactor loop.
- [PyZMQ releases](https://pypi.org/project/pyzmq/): PyZMQ and libzmq have separate versions. `zmq-vision doctor` reports both.
- [timm data configuration](https://huggingface.co/docs/timm/main/en/reference/data): model-specific evaluation transforms.
- [Keras MobileNet](https://keras.io/api/applications/mobilenet/): preprocessing and model constraints.
