#!/usr/bin/env python3
"""
Convenience launcher for the DR Design Pipeline web UI.

Usage:
    python serve.py           # port 8000
    python serve.py --port 8080
    python serve.py --open    # auto-open browser
"""
import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8000)
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--open", action="store_true", help="Open browser automatically")
p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
args = p.parse_args()

url = f"http://localhost:{args.port}"
print(f"\n  DR Design Pipeline  ·  {url}\n")

if args.open:
    import threading, time, webbrowser
    def _open():
        time.sleep(2)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

import uvicorn
uvicorn.run(
    "app:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
    log_level="warning",
)
