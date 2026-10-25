# Protocol and execution model

Each client owns a DEALER socket and sends one UTF-8 JSON frame. The server's ROUTER receives the connection identity followed by that frame. Only the network thread touches the ROUTER; a bounded `ThreadPoolExecutor` decodes images and executes models. Completed futures return through the same network thread. There is no in-process socket proxy.

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

`elapsed_ms` measures server processing and image serialization, excluding admission queueing and network round trips. Metadata is reserved for custom callers; the built-in predictors do not consume it. Requests must be JSON objects with a nonempty string ID (maximum 128 characters), dictionary metadata, and a string image payload. Omitting an ID generates a new one for older callers. Payloads must contain strict base64 and a Pillow-readable image.

Malformed JSON or image data produces an error response without terminating the server. Invalid multipart envelopes receive an error with ID `unknown`. Invalid IDs and oversized frames may be impossible to correlate; clients eventually time out. A custom caller should always send and check an ID.

## Limits and lifecycle

`ServerConfig` defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_workers` | 1 | Concurrent model calls |
| `max_pending_requests` | 16 | Accepted running plus queued futures |
| `max_message_bytes` | 8 MiB | libzmq frame limit and application size check |
| `max_image_pixels` | 16,000,000 | Decoded image width × height |
| `poll_timeout_ms` | 50 | Poll interval for shutdown and completed futures |

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

Text generation is outside this protocol. A future LLM backend would need explicit text messages, token budgets, cancellation, and potentially streaming responses rather than overloading the image payload.

## Upstream references

- [ZeroMQ thread-safety rules](https://zeromq.github.io/libzmq/zmq.html): contexts may be shared; ordinary sockets require thread ownership.
- [PyZMQ asyncio on Windows](https://github.com/zeromq/pyzmq/issues/1830): the selector bridge supports the default Proactor loop.
- [PyZMQ releases](https://pypi.org/project/pyzmq/): PyZMQ and libzmq have separate versions. `zmq-vision doctor` reports both.
- [timm data configuration](https://huggingface.co/docs/timm/main/en/reference/data): model-specific evaluation transforms.
- [Keras MobileNet](https://keras.io/api/applications/mobilenet/): preprocessing and model constraints.
