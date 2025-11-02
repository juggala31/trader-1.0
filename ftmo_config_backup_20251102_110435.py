# FTMO Trading System Configuration
import os

class FTMOConfig:
    # Trading Parameters
    SYMBOLS = ["US30", "US100", "UAX"]
    DEFAULT_SYMBOL = "US30"
    
    # Risk Management
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    SL_ATR_MULTIPLIER = 1.2
    TP_ATR_MULTIPLIER = 2.0
    MAX_CONCURRENT_TRADES = 3
    
    # FTMO Challenge Rules
    FTMO_200K_RULES = {
        'daily_loss_limit': 5000,
        'max_drawdown_limit': 10000, 
        'profit_target': 10000,
        'min_trading_days': 5,
        'max_trading_period': 30
    }
    
    # ML Model Parameters
    XGBOOST_CONFIDENCE_THRESHOLD = 0.6
    FALLBACK_TRIGGER_WIN_RATE = 0.35
    MODEL_RETRAIN_INTERVAL = 24  # hours
    
    # ZMQ Configuration
    ZMQ_PORTS = {
        'strategy_bus': 5556,
        'performance_bus': 5557, 
        'metrics_bus': 5558,
        'health_bus': 5559,
        'status_bus': 5560
    }
    
    # Service Configuration
    SERVICES = {
        'xgboost_strategy': {
            'process_name': 'python',
            'health_port': 5571,
            'memory_limit': 800
        },
        'fallback_strategy': {
            'process_name': 'python', 
            'health_port': 5572,
            'memory_limit': 400
        },
        'gui_dashboard': {
            'process_name': 'python',
            'health_port': 5573, 
            'memory_limit': 600
        },
        'zmq_bus': {
            'process_name': 'python',
            'health_port': 5574,
            'memory_limit': 300
        }
    }
    
    # File Paths
    MODEL_PATH = "models/xgboost_model.pkl"
    DATA_PATH = "data/"
    LOG_PATH = "logs/"
    
    @staticmethod
    def get_symbol_config(symbol):
        """Get symbol-specific configuration"""
        configs = {
            "US30": {"point_value": 1.0, "spread": 2.5},
            "US100": {"point_value": 0.1, "spread": 1.5},
            "UAX": {"point_value": 1.0, "spread": 3.0}
        }
        return configs.get(symbol, configs["US30"])
