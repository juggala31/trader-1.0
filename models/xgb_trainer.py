# exosati_trader/models/xgb_trainer.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(
        "xgboost is required. Install with:  python -m pip install xgboost\n"
        f"Import error: {e}"
    )

from exosati_trader.features.common_features import (
    FeatureParams, LabelParams, featurize_and_label
)

# --- I/O helpers (detect time column like the test script) -------------------------
_TIME_CANDIDATES = ["time", "timestamp", "ts", "date", "datetime"]

def _detect_time_column(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    for k in _TIME_CANDIDATES:
        if k in cols:
            return cols[k]
    raise SystemExit(f"CSV missing a time column. Tried: {_TIME_CANDIDATES}. Found: {list(df.columns)}")

def read_ohlc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = _detect_time_column(df)

    def pick(name: str) -> str | None:
        for c in df.columns:
            if c.lower() == name:
                return c
        return None
    need = ["open","high","low","close"]
    missing = [n for n in need if pick(n) is None]
    if missing:
        raise SystemExit(f"CSV missing OHLC columns: {missing}. Found: {list(df.columns)}")

    df = df.rename(columns={
        time_col: "time",
        pick("open"): "open",
        pick("high"): "high",
        pick("low"): "low",
        pick("close"): "close",
    })
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time","open","high","low","close"])
    return df.set_index("time").sort_index()


# --- Config dataclasses ------------------------------------------------------------
@dataclass(frozen=True)
class TrainParams:
    # label mapping: {-1,0,1} -> {0,1,2}
    max_depth: int = 6
    n_estimators: int = 400
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    # class weights to handle imbalance (for classes 0,1,2 mapped from -1,0,1)
    weight_neg1: float = 1.0
    weight_zero: float = 0.5
    weight_pos1: float = 1.0
    random_state: int = 42
    test_size: float = 0.2          # final holdout fraction (time-split aware option below)
    time_split: bool = True         # if True, split by time (head->train, tail->test); else sklearn random split

def _map_y_to_012(y: pd.Series) -> Tuple[np.ndarray, Dict[int, int]]:
    # {-1,0,1} -> {0,1,2}
    mapping = {-1: 0, 0: 1, 1: 2}
    y012 = y.map(mapping).to_numpy(dtype=np.int32, na_value=-1)
    # filter any nans that slipped through
    mask = (y012 >= 0)
    return y012[mask], mapping

def _compute_class_weights(y012: np.ndarray, p: TrainParams) -> np.ndarray:
    # Return a weight vector aligned to y012 (values in {0,1,2})
    w = np.ones_like(y012, dtype=np.float32)
    w[y012 == 0] = p.weight_neg1
    w[y012 == 1] = p.weight_zero
    w[y012 == 2] = p.weight_pos1
    return w


# --- Training core ----------------------------------------------------------------
def train_single(
    df: pd.DataFrame,
    feat_params: FeatureParams,
    lab_params: LabelParams,
    tparams: TrainParams,
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Fit a single XGB model on featurized data with labels.
    Returns: (model, X_train, X_test, metrics_dict)
    """
    X, ydf = featurize_and_label(df, feat_params, lab_params)
    y = ydf["y"]  # -1, 0, 1
    # map to 0/1/2 for softprob multi-class
    y012, mapping = _map_y_to_012(y)
    X = X.iloc[:len(y012)]  # safety align

    # time-aware split
    if tparams.time_split:
        n = len(X)
        n_train = int(n * (1 - tparams.test_size))
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y012[:n_train], y012[n_train:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y012, test_size=tparams.test_size, random_state=tparams.random_state, shuffle=True, stratify=y012
        )

    # weights
    w_train = _compute_class_weights(y_train, tparams)

    # model
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=tparams.max_depth,
        n_estimators=tparams.n_estimators,
        learning_rate=tparams.learning_rate,
        subsample=tparams.subsample,
        colsample_bytree=tparams.colsample_bytree,
        reg_lambda=tparams.reg_lambda,
        reg_alpha=tparams.reg_alpha,
        gamma=tparams.gamma,
        random_state=tparams.random_state,
        tree_method="hist",
        n_jobs=0,
        verbosity=1,
    )
    model.fit(X_train.to_numpy(), y_train, sample_weight=w_train)

    # metrics
    proba = model.predict_proba(X_test.to_numpy())
    y_pred = proba.argmax(axis=1)

    # macro AUC (one-vs-rest) if possible
    auc = None
    try:
        y_test_ovr = np.eye(3)[y_test]
        auc = roc_auc_score(y_test_ovr, proba, average="macro", multi_class="ovr")
    except Exception:
        pass

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "auc_macro_ovr": None if auc is None else float(auc),
        "report": report,
        "class_mapping": {-1: 0, 0: 1, 1: 2},
        "feature_columns": list(X.columns),
    }
    return model, X_train, X_test, metrics


# --- Artifact save/load ------------------------------------------------------------
def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def save_artifacts(
    outdir: Path,
    model: xgb.XGBClassifier,
    metrics: Dict,
    feat_params: FeatureParams,
    lab_params: LabelParams,
    tparams: TrainParams,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    tag = _timestamp()
    model_dir = outdir / f"xgb_{tag}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.json"
    model.save_model(str(model_path))

    # Save meta/config
    meta = {
        "created_utc": tag,
        "metrics": metrics,
        "feat_params": asdict(feat_params),
        "label_params": asdict(lab_params),
        "train_params": asdict(tparams),
        "model_file": "model.json",
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Save feature column order
    (model_dir / "feature_columns.json").write_text(json.dumps(metrics.get("feature_columns", []), indent=2))

    return model_dir

def load_artifacts(model_dir: Path) -> Tuple[xgb.XGBClassifier, Dict, list[str]]:
    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / "model.json"))
    meta = json.loads((model_dir / "meta.json").read_text())
    cols = json.loads((model_dir / "feature_columns.json").read_text())
    return model, meta, cols


# --- CLI --------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Exosati — XGBoost trainer (time-aware)")
    ap.add_argument("--data", default=r".\data\US30_1m.csv", help="OHLC CSV (time/timestamp/ts/date/datetime + open/high/low/close)")
    ap.add_argument("--outdir", default=r".\models\xgb", help="Where to save versioned models")
    # feature params
    ap.add_argument("--ema-fast", type=int, default=12)
    ap.add_argument("--ema-slow", type=int, default=26)
    ap.add_argument("--sma-fast", type=int, default=20)
    ap.add_argument("--sma-slow", type=int, default=50)
    ap.add_argument("--rsi", type=int, default=14)
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--adx", type=int, default=14)
    ap.add_argument("--vol-window", type=int, default=30)
    ap.add_argument("--atr-pctile-window", type=int, default=500)
    # label params
    ap.add_argument("--sl-atr", type=float, default=1.2)
    ap.add_argument("--tp-atr", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--intrabar", choices=["HL","OHLC","CLOSE"], default="HL")
    # train params
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--lambda", dest="reg_lambda", type=float, default=1.0)
    ap.add_argument("--alpha", dest="reg_alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--w-neg1", type=float, default=1.0)
    ap.add_argument("--w-zero", type=float, default=0.5)
    ap.add_argument("--w-pos1", type=float, default=1.0)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-time-split", action="store_true", help="Use random split instead of time-based")
    return ap

def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        return 2

    df = read_ohlc_csv(data_path)

    feat_params = FeatureParams(
        ema_fast=args.ema_fast, ema_slow=args.ema_slow,
        sma_fast=args.sma_fast, sma_slow=args.sma_slow,
        rsi_period=args.rsi, atr_period=args.atr, adx_period=args.adx,
        vol_window=args.vol_window, atr_pctile_window=args.atr_pctile_window,
    )
    lab_params = LabelParams(
        atr_sl=args.sl_atr, atr_tp=args.tp_atr,
        max_horizon=args.horizon, intrabar_exit=args.intrabar,
    )
    tparams = TrainParams(
        max_depth=args.max_depth, n_estimators=args.n_estimators,
        learning_rate=args.lr, subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha, gamma=args.gamma,
        weight_neg1=args.w_neg1, weight_zero=args.w_zero, weight_pos1=args.w_pos1,
        test_size=args.test_size, time_split=not args.no_time_split,
    )

    print("[INFO] Featurizing + labeling …")
    model, Xtr, Xte, metrics = train_single(df, feat_params, lab_params, tparams)

    print("\n=== METRICS (holdout) ===")
    print(json.dumps(metrics["report"], indent=2))
    if metrics.get("auc_macro_ovr") is not None:
        print(f"AUC (macro ovr): {metrics['auc_macro_ovr']:.4f}")
    print(f"Train size: {metrics['n_train']}, Test size: {metrics['n_test']}")

    outdir = Path(args.outdir)
    save_dir = save_artifacts(outdir, model, metrics, feat_params, lab_params, tparams)
    print(f"\n[OK] Saved model to: {save_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
