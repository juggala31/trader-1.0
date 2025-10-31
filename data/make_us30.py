from __future__ import annotations
import csv, random, os
from datetime import datetime, timedelta, timezone

out_path = os.path.abspath(r".\data\US30_1m.csv")
bars = 15000                 # >= train(10000) + test(2000) + buffer
start_price = 39000.0
ts = datetime.now(timezone.utc) - timedelta(minutes=bars)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

price = start_price
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ts","open","high","low","close"])
    for _ in range(bars):
        o = price
        step = random.gauss(0, 30.0)  # ~30-point std per minute
        h = o + abs(step)*0.6 + random.random()*5.0
        l = o - abs(step)*0.6 - random.random()*5.0
        c = o + step
        price = c
        ts = ts + timedelta(minutes=1)
        w.writerow([ts.isoformat(), f"{o:.6f}", f"{h:.6f}", f"{l:.6f}", f"{c:.6f}"])

print(f"Wrote {bars} bars to {out_path}")
