#!/usr/bin/env bash
# Serve the repo root over HTTP with the wasm MIME type and no caching.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8000}"

exec python3 - "$PORT" <<'PY'
import functools, http.server, os, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

port = int(sys.argv[1])
handler = functools.partial(Handler, directory=os.getcwd())
print(f"http://localhost:{port}/", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
PY
