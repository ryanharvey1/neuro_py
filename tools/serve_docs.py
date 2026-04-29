"""Serve the built docs site with browser-friendly MIME types."""

from __future__ import annotations

import argparse
import mimetypes
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    """Run a local static docs server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument(
        "--directory",
        default="site",
        help="Built docs directory to serve.",
    )
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("application/wasm", ".wasm")

    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {directory} at http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
