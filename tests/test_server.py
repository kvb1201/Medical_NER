#!/usr/bin/env python3
"""HTTP behaviour of app/server.py, using a stub model so no checkpoints are needed."""

import importlib.util
import json
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

SERVER_PATH = Path(__file__).resolve().parents[1] / "app" / "server.py"


class _StubModel:
    def predict(self, sentences):
        return [["B-DISEASE" if tok == "asthma" else "O" for tok in sent] for sent in sentences]


@pytest.fixture()
def server_url(monkeypatch):
    spec = importlib.util.spec_from_file_location("clinical_ner_server", SERVER_PATH)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    monkeypatch.setattr(server, "_models", [("STUB", _StubModel())])
    monkeypatch.setattr(server, "_model_error", None)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address
    httpd.shutdown()
    httpd.server_close()


def _raw_request(address, head: bytes, body: bytes = b"") -> tuple:
    """Send raw bytes and return (status_code, parsed_json_or_None)."""
    with socket.create_connection(address, timeout=5) as sock:
        sock.sendall(head + body)
        sock.shutdown(socket.SHUT_WR)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    status = int(data.split(b" ", 2)[1])
    payload = data.split(b"\r\n\r\n", 1)[1]
    return status, (json.loads(payload) if payload.startswith(b"{") else None)


def _post(address, body: bytes, content_length: str = None):
    length = str(len(body)) if content_length is None else content_length
    head = (f"POST /api/extract HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
            f"Content-Length: {length}\r\n\r\n").encode()
    return _raw_request(address, head, body)


def test_extract_returns_entities(server_url):
    status, data = _post(server_url, json.dumps({"text": "She has asthma."}).encode())
    assert status == 200
    assert data["models"][0]["entities"] == [{"text": "asthma", "type": "DISEASE", "start": 8, "end": 14}]


@pytest.mark.parametrize("content_length", ["abc", "-1", "1.5", "²"])
def test_bad_content_length_gets_400_not_a_crash_or_hang(server_url, content_length):
    status, data = _post(server_url, b'{"text":"x"}', content_length=content_length)
    assert status == 400
    assert "Content-Length" in data["error"]


def test_invalid_utf8_gets_400(server_url):
    status, data = _post(server_url, b"\xff\xfe")
    assert status == 400


def test_oversized_body_gets_413(server_url):
    status, _ = _post(server_url, b"", content_length="100001")
    assert status == 413


def test_favicon_is_served(server_url):
    status, _ = _raw_request(server_url, b"GET /favicon.ico HTTP/1.1\r\nHost: test\r\n\r\n")
    assert status == 200
