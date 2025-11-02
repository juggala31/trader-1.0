# FTMO Final Startup - Handles missing model files gracefully
import sys
import time
import logging

def setup_logging():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ftmo_trading.log'),
            logging.StreamHandler()
        ]
    )

def main():
    print("🚀 FTMO Trading System - Final Startup")
    print("======================================")
    
    setup_logging()
    
    try:
        from ftmo_integrated_system import FTMOIntegratedSystem
        
        print("Initializing system...")
        system = FTMOIntegratedSystem(account_id="1600038177", challenge_type="200k")
        
        # Check model status
        model_info = system.model_loader.get_model_info()
        if not model_info['loaded']:
            print("⚠️  Using fallback mode - XGBoost model not available")
            print("   Train your model and save it as: models/xgboost_model.pkl")
        else:
            print("✅ XGBoost model loaded successfully")
        
        # Show system status
        status = system.get_system_status()
        print(f"\nSystem Status:")
        print(f"• Current Strategy: {status['current_strategy']}")
        print(f"• Model Loaded: {model_info['loaded']}")
        print(f"• FTMO Target: ${status['ftmo_metrics']['profit_target']}")
        print(f"• Health Status: {status['health_status']['overall_status']}")
        
        print("\nStarting in 3 seconds...")
        print("Press Ctrl+C to stop")
        time.sleep(3)
        
        # Start trading
        system.start_trading()
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        logging.error(f"Startup failed: {e}")
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if MetaTrader is running")
        print("2. Verify OANDA demo account credentials")
        print("3. Check port conflicts with: python port_manager.py")

if __name__ == "__main__":
    main()
