# FTMO Trading Configuration
class FTMOConfig:
    # Trading Parameters
    SYMBOLS = ["US30", "US100", "UAX"]
    DEFAULT_SYMBOL = "US30"
    TIMEFRAME = "M15"  # M1, M5, M15, H1, H4, D1
    
    # Risk Management
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_DAILY_TRADES = 10
    MAX_CONCURRENT_TRADES = 3
    
    # Stop Loss / Take Profit
    SL_ATR_MULTIPLIER = 1.2
    TP_ATR_MULTIPLIER = 2.0
    MAX_SL_PERCENT = 2.0  # Maximum 2% stop loss
    
    # FTMO Challenge Rules
    FTMO_200K_RULES = {
        'starting_balance': 200000,
        'daily_loss_limit': 5000,
        'max_drawdown_limit': 10000,
        'profit_target': 10000,
        'min_trading_days': 5,
        'max_trading_period': 30
    }
    
    # Strategy Parameters
    XGBOOST_CONFIDENCE_THRESHOLD = 0.6
    FALLBACK_TRIGGER_WIN_RATE = 0.35
    STRATEGY_SWITCH_COOLDOWN = 3600  # 1 hour in seconds
    
    # MetaTrader5 Settings
    MT5_SETTINGS = {
        'path': "C:\\\\Program Files\\\\MetaTrader 5\\\\terminal64.exe",
        'server': "YourBrokerServer",
        'login': 1600038177,
        'password': "YourPassword"
    }
    
    # ZMQ Configuration
    ZMQ_PORTS = {
        'strategy_bus': 5556,
        'performance_bus': 5557,
        'metrics_bus': 5558,
        'health_bus': 5559
    }

# Trading Hours Configuration
TRADING_HOURS = {
    'US30': {'start': 9, 'end': 16},  # EST hours
    'US100': {'start': 9, 'end': 16},
    'UAX': {'start': 9, 'end': 16}
}

# Symbol Configuration
SYMBOL_CONFIG = {
    "US30": {
        "point_value": 1.0,
        "spread": 2.5,
        "stops_level": 10,
        "volume_min": 0.1,
        "volume_max": 100.0
    },
    "US100": {
        "point_value": 0.1,
        "spread": 1.5,
        "stops_level": 10,
        "volume_min": 0.1,
        "volume_max": 100.0
    },
    "UAX": {
        "point_value": 1.0,
        "spread": 3.0,
        "stops_level": 15,
        "volume_min": 0.1,
        "volume_max": 50.0
    }
}
