from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class BrokerCfg:
    platform: str
    server: str
    login: int
    password: str

@dataclass
class RiskCfg:
    max_daily_loss: float
    max_total_dd: float
    per_trade_risk_pct: float
    trade_session: dict

@dataclass
class PortsCfg:
    health: int = 5559

@dataclass
class PathsCfg:
    logs: str = "logs"
    exports: str = "exports"

@dataclass
class AppConfig:
    broker: BrokerCfg
    risk: RiskCfg
    symbols: list[str]
    timeframes: list[str]
    ports: PortsCfg
    paths: PathsCfg

def load(config_path: str | Path) -> AppConfig:
    p = Path(config_path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        broker=BrokerCfg(**raw["broker"]),
        risk=RiskCfg(**raw["risk"]),
        symbols=list(raw.get("symbols", [])),
        timeframes=list(raw.get("timeframes", [])),
        ports=PortsCfg(**raw.get("ports", {})),
        paths=PathsCfg(**raw.get("paths", {})),
    )