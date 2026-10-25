"""Client misuse fails before transport or image handling begins."""

import pytest

from zmq_inference.client.sync_client import SyncClient


def test_infer_requires_connection(tiny_pil_image):
    with pytest.raises(RuntimeError, match="not connected"):
        SyncClient().infer(tiny_pil_image)


def test_ping_requires_connection():
    with pytest.raises(RuntimeError, match="not connected"):
        SyncClient().ping()


def test_disconnect_is_idempotent():
    client = SyncClient()
    client.disconnect()
    client.disconnect()
