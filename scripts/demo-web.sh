#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
demo_python="${ZMQ_DEMO_PYTHON:-python3}"
if [[ -z "${ZMQ_DEMO_PYTHON:-}" && -z "${VIRTUAL_ENV:-}" && -x "$repo/.venv/bin/python" ]]; then
    demo_python="$repo/.venv/bin/python"
fi

options=(--demo)
case "${1:-}" in
    --connect) shift; options=() ;;
    --vlm) options=() ;;
esac

export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$demo_python" -m zmq_inference web "${options[@]}" "$@"
