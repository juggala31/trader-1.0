from __future__ import annotations
import subprocess
import sys
import time
import os

def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    py = sys.executable
    env = os.environ.copy()

    # 1) Start healthbeat in background
    hb = subprocess.Popen([py, "-m", "exosati_trader.services.healthbeat"], cwd=root, env=env)
    print(f"[start_all] healthbeat pid={hb.pid}")
    time.sleep(0.8)

    # 2) Prefer YOUR enhanced_trading_dashboard.py if it exists
    target_py = os.path.join(root, "enhanced_trading_dashboard.py")
    if os.path.exists(target_py):
        cmd = [py, target_py]
        print("[start_all] launching enhanced_trading_dashboard.py")
    else:
        # fallback to package GUI only if your file is missing
        cmd = [py, "-m", "exosati_trader.app.gui_entry"]
        print("[start_all] launching package gui_entry (fallback)")

    rc = 1
    try:
        rc = subprocess.call(cmd, cwd=root, env=env)
    finally:
        # 3) Teardown healthbeat
        try:
            hb.terminate()
            try:
                hb.wait(timeout=3)
            except Exception:
                hb.kill()
        except Exception:
            pass

    sys.exit(rc)

if __name__ == "__main__":
    main()