"""Regenerate original SVG artwork and a captured terminal demo."""

import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    output = subprocess.check_output([sys.executable, "-m", "zmq_inference", "demo"], text=True)
    (ASSETS / "demo-output.txt").write_text(output)
    lines = ["$ zmq-vision demo", "", *output.rstrip().splitlines()]
    rows = "\n".join(
        f'<text x="36" y="{100 + index * 32}" fill="{color}">{escape(line)}</text>'
        for index, line in enumerate(lines)
        for color in ["#5eead4" if index == 0 or line.startswith("PASS") else "#dbeafe"]
    )
    (
        ASSETS / "demo.svg"
    ).write_text(f"""<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="400" viewBox="0 0 1040 400" role="img" aria-labelledby="title desc">
<title id="title">ZMQ Vision: captured terminal demo</title>
<desc id="desc">A real local TCP run validates health, synchronous red-image inference, asynchronous blue-image inference, and shutdown.</desc>
<rect width="1040" height="400" rx="18" fill="#0b1220"/>
<path d="M18 0h1004a18 18 0 0 1 18 18v36H0V18A18 18 0 0 1 18 0" fill="#172136"/>
<circle cx="28" cy="27" r="6" fill="#fb7185"/><circle cx="49" cy="27" r="6" fill="#fbbf24"/><circle cx="70" cy="27" r="6" fill="#34d399"/>
<text x="520" y="33" text-anchor="middle" fill="#94a3b8" font-size="14" font-family="sans-serif">zmq-vision / local TCP / no model downloads</text>
<g font-family="'DejaVu Sans Mono',Consolas,monospace" font-size="17">{rows}</g>
<text x="36" y="374" fill="#64748b" font-size="13" font-family="sans-serif">Captured output · processing time varies by machine · RGB statistics demo backend</text>
</svg>""")
    (
        ASSETS / "banner.svg"
    ).write_text("""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360" viewBox="0 0 1200 360" role="img" aria-labelledby="title desc">
<title id="title">ZMQ Vision</title><desc id="desc">Image models, one messaging interface. Sync and async clients connect to optional PyTorch, TensorFlow, or custom backends.</desc>
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0b1220"/><stop offset="1" stop-color="#172554"/></linearGradient></defs>
<rect width="1200" height="360" rx="24" fill="url(#bg)"/>
<g stroke="#263657" stroke-width="1" opacity=".55"><path d="M720 0v360m80-360v360m80-360v360m80-360v360m80-360v360m80-360v360M680 60h520M680 140h520M680 220h520M680 300h520"/></g>
<text x="60" y="74" fill="#5eead4" font-family="sans-serif" font-size="14" letter-spacing="4">COMPUTER VISION / ZEROMQ</text>
<text x="56" y="158" fill="#f8fafc" font-family="sans-serif" font-size="72" font-weight="700">ZMQ Vision</text>
<text x="60" y="208" fill="#cbd5e1" font-family="sans-serif" font-size="24">Image models. One messaging interface.</text>
<text x="60" y="277" fill="#94a3b8" font-family="sans-serif" font-size="17">Python library · sync + async · optional ML backends</text>
<text x="60" y="315" fill="#64748b" font-family="sans-serif" font-size="14">BY JAMAL SAEEDI</text>
<g fill="none" stroke="#5eead4" stroke-width="3"><rect x="766" y="95" width="116" height="116" rx="16"/><path d="M784 191l29-36 22 22 18-13 13 27z"/><circle cx="850" cy="122" r="9"/><path d="M890 152h74m-12-9 12 9-12 9"/><rect x="976" y="105" width="150" height="96" rx="14"/></g>
<text x="1051" y="148" text-anchor="middle" fill="#f8fafc" font-family="sans-serif" font-size="19">MODEL</text>
<text x="1051" y="176" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="13">worker pool</text>
<text x="824" y="245" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">IMAGE</text>
<text x="1051" y="245" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">PREDICTIONS</text>
</svg>""")
    print("Rendered", ASSETS)


if __name__ == "__main__":
    main()
