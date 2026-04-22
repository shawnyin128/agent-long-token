from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request

from agentdiet.config import get_config


def check_once(url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            code = resp.status
            if code == 200:
                return True, f"200 OK {url}"
            return False, f"{code} {url}"
    except urllib.error.URLError as e:
        return False, f"{type(e).__name__}: {e} ({url})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e} ({url})"


def wait_until_healthy(url: str, timeout_s: float, poll_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    last_msg = ""
    while time.monotonic() < deadline:
        ok, msg = check_once(url)
        last_msg = msg
        if ok:
            print(msg, file=sys.stderr)
            return True
        time.sleep(poll_s)
    print(f"timeout after {timeout_s}s; last={last_msg}", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Poll vLLM /models endpoint until ready")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--url", default=None, help="override health URL; default derived from config")
    args = parser.parse_args(argv)

    cfg = get_config()
    url = args.url or f"{cfg.base_url.rstrip('/')}/models"
    ok = wait_until_healthy(url, timeout_s=args.timeout, poll_s=args.poll)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
