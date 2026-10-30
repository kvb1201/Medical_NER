#!/usr/bin/env python3
"""Serve the clinical NER page and tag notes with the repo's HMM, MEMM, and CRF.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python app/server.py

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

import joblib  # noqa: E402

from src.infer import extract_text  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000
# Same filenames written by src/models/train_and_evaluate.py
MODEL_NAMES = ("hmm", "memm", "crf")
MODELS_DIR = ROOT / "models"
PAGE = Path(__file__).resolve().parent / "static" / "index.html"

_models = None
_model_error = None
_lock = threading.Lock()


def get_models():
    """Load HMM, MEMM, and CRF once. Returns (models, error_message)."""
    global _models, _model_error
    if _models is not None or _model_error is not None:
        return _models, _model_error
    with _lock:
        if _models is None and _model_error is None:
            missing = [name for name in MODEL_NAMES if not (MODELS_DIR / f"{name}.joblib").exists()]
            if missing:
                _model_error = (
                    "Missing checkpoints: "
                    + ", ".join(f"models/{name}.joblib" for name in missing)
                    + ". Create them with PYTHONPATH=. .venv/bin/python src/models/train_and_evaluate.py "
                    "and restart this server."
                )
            else:
                _models = [
                    (name.upper(), joblib.load(MODELS_DIR / f"{name}.joblib"))
                    for name in MODEL_NAMES
                ]
    return _models, _model_error


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            self._json(404, {"error": "Not found."})
            return
        self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/extract":
            self._json(404, {"error": "Not found."})
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 100_000:
            self._json(413, {"error": "Request is too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Send a JSON object with a text field."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("text", ""), str):
            self._json(400, {"error": "The text field must be a string."})
            return

        models, error = get_models()
        if error:
            self._json(503, {"error": error})
            return
        text = payload.get("text", "")
        try:
            with _lock:
                readings = [extract_text(text, model, name) for name, model in models]
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception:
            self._json(500, {"error": "The tagger failed on this note."})
            return
        self._json(200, {"text": text, "models": readings})


def main() -> None:
    get_models()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Clinical NER is at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
