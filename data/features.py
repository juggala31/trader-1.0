# exosati_trader/data/features.py
# Wilder's ATR utilities: batch (pandas) + streaming (online) calculators.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import math

try:
    import pandas as pd  # type: ignore
    import numpy as np   # type: ignore
except Exception:  # pragma: no cover - pandas/numpy may be absent in minimal envs
    pd = None
    np = None


def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    """
    True Range (TR) = max(
        high - low,
        abs(high - prev_close),
        abs(low  - prev_close)
    )
    If prev_close is None (first bar), TR falls back to high - low.
    """
    hl = high - low
    if prev_close is None or math.isnan(prev_close):
        return max(0.0, hl)
    return max(hl, abs(high - prev_close), abs(low - prev_close))


@dataclass
class ATRState:
    period: int = 14
    count: int = 0                 # how many TR samples we've ingested
    last_close: Optional[float] = None
    atr: Optional[float] = None    # Wilder ATR (RMA)
    tr_sum: float = 0.0            # used to seed with SMA of first N TRs

    def reset(self) -> None:
        self.count = 0
        self.last_close = None
        self.atr = None
        self.tr_sum = 0.0

    @property
    def ready(self) -> bool:
        """True when period samples have been incorporated and ATR is rolling."""
        return self.atr is not None and self.count >= self.period


class ATRCalculator:
    """
    Online (streaming) Wilder ATR calculator.

    Usage:
        atr = ATRCalculator(period=14)
        value = atr.update(high, low, close)   # returns None until warmed up
        if atr.ready:
            print(atr.value)
    """
    def __init__(self, period: int = 14):
        if period <= 1:
            # Wilder's ATR requires N >= 2 (seed with first N TRs)
            raise ValueError("ATR period must be >= 2")
        self._state = ATRState(period=period)

    @property
    def period(self) -> int:
        return self._state.period

    @property
    def value(self) -> Optional[float]:
        return self._state.atr

    @property
    def ready(self) -> bool:
        return self._state.ready

    @property
    def count(self) -> int:
        return self._state.count

    def reset(self) -> None:
        self._state.reset()

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """
        Ingest a completed bar's H, L, C and update internal ATR.
        Returns the current ATR value (or None if not yet warmed).
        """
        s = self._state
        tr = _true_range(high, low, s.last_close)

        # Accumulate until we have period samples to seed SMA(TR)
        if s.count < s.period:
            s.tr_sum += tr
            s.count += 1
            s.last_close = close
            if s.count == s.period:
                s.atr = s.tr_sum / s.period  # seed with SMA of first N TRs
            return s.atr  # will be None until just seeded

        # Wilder RMA update: ATR_t = (ATR_{t-1}*(n-1) + TR_t) / n
        s.atr = ((s.atr * (s.period - 1)) + tr) / s.period  # type: ignore
        s.count += 1
        s.last_close = close
        return s.atr


class RollingIndicators:
    """
    Convenience manager for per-symbol ATR calculators.
    """
    def __init__(self, period: int = 14):
        self.period = period
        self._atr_by_symbol: Dict[str, ATRCalculator] = {}

    def atr(self, symbol: str) -> ATRCalculator:
        calc = self._atr_by_symbol.get(symbol)
        if calc is None:
            calc = ATRCalculator(period=self.period)
            self._atr_by_symbol[symbol] = calc
        return calc

    def reset_symbol(self, symbol: str) -> None:
        if symbol in self._atr_by_symbol:
            self._atr_by_symbol[symbol].reset()

    def reset_all(self) -> None:
        for calc in self._atr_by_symbol.values():
            calc.reset()


# -----------------------
# Batch/Pandas utilities
# -----------------------

def compute_atr(df, period: int = 14, high_col: str = "high",
                low_col: str = "low", close_col: str = "close",
                out_col: str = "atr"):

    """
    Vectorized Wilder ATR over a pandas DataFrame with columns:
    [high_col, low_col, close_col], indexed by time (any frequency).

    Returns the same DataFrame with a new column out_col (ATR).
    The first `period` ATR value equals SMA(TR) over the first `period` rows,
    and thereafter uses Wilder's recursive RMA.

    Notes:
      - Requires pandas and numpy. If not available, raises ImportError.
      - NaN will appear for the initial rows until the seed is available.
    """
    if pd is None or np is None:
        raise ImportError("pandas/numpy are required for compute_atr")

    req = [high_col, low_col, close_col]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"compute_atr: missing column '{c}'")

    highs = df[high_col].astype(float).values
    lows = df[low_col].astype(float).values
    closes = df[close_col].astype(float).values

    n = len(df)
    atr = np.full(n, np.nan, dtype=float)

    # True Range series
    prev_close = np.roll(closes, 1)
    prev_close[0] = np.nan
    hl = highs - lows
    hc = np.abs(highs - prev_close)
    lc = np.abs(lows - prev_close)
    tr = np.maximum.reduce([hl, hc, lc])
    tr[0] = max(0.0, highs[0] - lows[0])  # first bar fallback

    if n == 0:
        df[out_col] = atr
        return df

    if period <= 1:
        raise ValueError("ATR period must be >= 2")

    if n >= period:
        seed = np.nanmean(tr[:period])
        atr[period - 1] = seed
        # Wilder RMA
        for i in range(period, n):
            atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period

    # assign
    df[out_col] = atr
    return df
