# rt_safe_helpers.py
# Drop-in helpers to avoid common runtime crashes.

from __future__ import annotations
from typing import Any, Optional, Sequence
import numpy as np

def is_empty(arr: Any) -> bool:
    \"\"\"Return True if arr is None or has zero length/size.\"\"\"
    if arr is None:
        return True
    try:
        # numpy arrays
        a = np.asarray(arr)
        return a.size == 0
    except Exception:
        # generic sequences
        try:
            return len(arr) == 0  # type: ignore[arg-type]
        except Exception:
            return False

def safe_any(mask: Any) -> bool:
    \"\"\"Truthiness for arrays: True if any element is True/nonzero.\"\"\"
    try:
        return bool(np.any(mask))
    except Exception:
        try:
            # fallback for Python lists
            return any(bool(x) for x in mask)  # type: ignore
        except Exception:
            return bool(mask)

def guard_market_data(data: Any) -> Optional[np.ndarray]:
    \"\"\"Return a 1D numpy array or None if data is unusable.\"\"\"
    if is_empty(data):
        return None
    arr = np.asarray(data)
    if arr.ndim == 0:
        # single scalar -> make it 1-element
        arr = arr.reshape(1,)
    elif arr.ndim > 1:
        # take last row/col if someone passed a frame
        arr = arr.reshape(-1)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return arr

def mt5_result_ok(result: Any) -> bool:
    \"\"\"True if an MT5 order_send-like result indicates success.\"\"\"
    # MetaTrader5.PY returns an object with 'retcode'. 10009/10008 are OK in many cases.
    # We defensively check for presence and pack success codes here.
    if result is None:
        return False
    rc = getattr(result, 'retcode', None)
    return rc in (10009, 10008)

def mt5_result_errmsg(result: Any) -> str:
    if result is None:
        return "order_send returned None (connection or parameter issue)"
    rc = getattr(result, 'retcode', None)
    return f"retcode={rc}, comment={getattr(result, 'comment', '')}, request={getattr(result, 'request', None)}"