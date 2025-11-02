# FTMO Working Startup - Simple version
print("FTMO Trading System - Starting...")
print("==================================")

try:
    # Basic imports
    from strategy_orchestrator import StrategyOrchestrator
    from ftmo_challenge_logger import FTMOChallengeLogger
    
    # Simple config
    class SimpleConfig:
        ZMQ_PORTS = {'strategy_bus': 5556}
        MODEL_PATH = "models/xgboost_model.pkl"
        
    print("Initializing components...")
    
    # Initialize with simple config
    config = SimpleConfig()
    orchestrator = StrategyOrchestrator(config)
    logger = FTMOChallengeLogger("1600038177", "200k")
    
    print("✅ Components initialized successfully")
    print(f"Current strategy: {orchestrator.current_strategy}")
    print(f"Account: 1600038177")
    print(f"Challenge: 200k")
    
    # Test a trade
    logger.log_trade({'profit': 50, 'symbol': 'US30'})
    print(f"Test trade logged: ${logger.metrics['total_profit']} profit")
    
    print("`nSystem is ready for integration with your existing trading logic!")
    print("Next: Connect this to your actual MetaTrader5 trading system.")
    
except Exception as e:
    print(f"Error: {e}")
    print("This is normal if your existing system is already running.")
    print("Try stopping your current system first.")
