"""Install a built wheel into a fresh venv and smoke-test outside the checkout."""

import email
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


def main():
    (wheel,) = Path(sys.argv[1] if len(sys.argv) > 1 else "dist").resolve().glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (metadata_file,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        metadata = email.message_from_bytes(archive.read(metadata_file))
        assert metadata["Name"] == "zmq-vision"
        assert "zmq_inference/py.typed" in archive.namelist()
        labels = json.loads(archive.read("zmq_inference/data/imagenet_classes.json"))
        assert labels["0"] and labels["999"]
    with tempfile.TemporaryDirectory(prefix="zmq-vision-wheel-") as temporary:
        root = Path(temporary)
        if importlib.util.find_spec("ensurepip") is None and shutil.which("uv"):
            subprocess.run(["uv", "venv", "--seed", str(root / "venv")], check=True)
        else:
            venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", str(wheel)], check=True, cwd=root)
        result = subprocess.check_output(
            [str(python), "-m", "zmq_inference", "demo", "--json"], text=True, cwd=root
        )
        assert json.loads(result)["health"] == "ok"
        subprocess.run(
            [
                str(python),
                "-c",
                "from zmq_inference.utils.labels import load_imagenet_labels; "
                "import importlib.util; "
                "assert len(load_imagenet_labels()) >= 1000; "
                "assert importlib.util.find_spec('torch') is None; "
                "assert importlib.util.find_spec('tensorflow') is None; "
                "assert importlib.util.find_spec('numpy') is None",
            ],
            check=True,
            cwd=root,
        )
        command_dir = python.parent
        for name in ("zmq-vision", "zmq-inference"):
            command = command_dir / (name + (".exe" if sys.platform == "win32" else ""))
            subprocess.run([str(command), "--version"], check=True, cwd=root)
    print(f"PASS: {wheel.name}; clean install, bundled labels, both CLIs, sync/async TCP demo")


if __name__ == "__main__":
    main()
