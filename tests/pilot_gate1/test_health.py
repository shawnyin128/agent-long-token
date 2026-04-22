from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentdiet.cli.health import check_once, main, wait_until_healthy


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def live_server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def test_check_once_200(live_server):
    ok, msg = check_once(f"{live_server}/v1/models")
    assert ok, msg


def test_wait_until_healthy_success(live_server):
    assert wait_until_healthy(f"{live_server}/v1/models", timeout_s=5.0, poll_s=0.1)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_wait_until_healthy_timeout_on_closed_port():
    port = _free_port()
    start = time.monotonic()
    ok = wait_until_healthy(f"http://127.0.0.1:{port}/v1/models", timeout_s=0.5, poll_s=0.1)
    assert ok is False
    assert time.monotonic() - start < 5.0


def test_main_default_url_derived_from_config(live_server, monkeypatch):
    monkeypatch.setenv("AGENTDIET_BASE_URL", f"{live_server}/v1")
    rc = main(["--timeout", "3", "--poll", "0.1"])
    assert rc == 0


def test_main_returns_1_on_timeout():
    port = _free_port()
    rc = main(["--timeout", "0.5", "--poll", "0.1", "--url", f"http://127.0.0.1:{port}/v1/models"])
    assert rc == 1
