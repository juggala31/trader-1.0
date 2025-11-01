# Fixed Integration Test - Simple version
import time
import sys  # Added this import

class SimpleIntegrationTest:
    def run_tests(self):
        print("FTMO SIMPLE INTEGRATION TEST")
        print("=============================")
        
        # Test 1: Basic imports
        try:
            from strategy_orchestrator import StrategyOrchestrator
            print("✅ StrategyOrchestrator imports correctly")
        except Exception as e:
            print(f"❌ StrategyOrchestrator import failed: {e}")
            return False
            
        # Test 2: FTMO Logger
        try:
            from ftmo_challenge_logger import FTMOChallengeLogger
            print("✅ FTMOChallengeLogger imports correctly")
        except Exception as e:
            print(f"❌ FTMOChallengeLogger import failed: {e}")
            return False
            
        # Test 3: Basic functionality
        try:
            # Simple config
            class TestConfig:
                ZMQ_PORTS = {'test': 5556}
                
            config = TestConfig()
            orchestrator = StrategyOrchestrator(config)
            print(f"✅ Strategy orchestrator created: {orchestrator.current_strategy}")
            
            # Test FTMO logger
            logger = FTMOChallengeLogger("test", "200k")
            logger.log_trade({'profit': 100})
            print(f"✅ FTMO logger working: {logger.metrics['total_trades']} trades")
            
        except Exception as e:
            print(f"❌ Functionality test failed: {e}")
            return False
            
        print("🎉 All tests passed! System is ready.")
        return True

if __name__ == "__main__":
    tester = SimpleIntegrationTest()
    success = tester.run_tests()
    sys.exit(0 if success else 1)
