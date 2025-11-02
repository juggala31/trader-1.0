from __future__ import annotations
import time, json, threading
from pathlib import Path

DEFAULT_FILE = Path('logs/healthbeat.json')

def run_heartbeat(interval: float = 2.0, outfile: str | Path = DEFAULT_FILE):
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        payload = {
            "service": "healthbeat",
            "ts": time.time(),
            "status": "ok"
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        time.sleep(interval)

def main():
    run_heartbeat()

if __name__ == '__main__':
    main()