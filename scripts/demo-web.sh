#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
demo_python="${ZMQ_DEMO_PYTHON:-python3}"
if [[ -z "${ZMQ_DEMO_PYTHON:-}" && -z "${VIRTUAL_ENV:-}" && -x "$repo/.venv/bin/python" ]]; then
    demo_python="$repo/.venv/bin/python"
fi

options=(--demo)
if [[ "${1:-}" == "--connect" ]]; then
    shift
    options=()
fi

export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$demo_python" -m zmq_inference web "${options[@]}" "$@"
