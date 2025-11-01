from market_regime_detector import MarketRegimeDetector, RegimeAwareTrading
import pandas as pd
import numpy as np

def test_basic_functionality():
    print("Testing Basic Market Regime Detection Functionality...")
    
    # Test 1: Basic regime detection
    print("\n1. Testing Market Regime Detector...")
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    price_series = pd.Series(prices, index=dates)
    
    detector = MarketRegimeDetector()
    detector.train_model(price_series)
    regime_pred = detector.predict_regime(price_series)
    
    print(f"✓ Market Regime: {regime_pred['regime_label']}")
    print(f"✓ Confidence: {regime_pred['confidence']:.2%}")
    
    # Test 2: Regime statistics
    print("\n2. Testing Regime Statistics...")
    stats = detector.get_regime_statistics(price_series)
    for regime, data in stats.items():
        if data:  # Check if data exists
            count = data.get('count', 0)
            mean_return = data.get('mean_return', 0)
            volatility = data.get('volatility', 0)
            print(f"✓ Regime {regime}: {count} samples, Return: {mean_return:.4%}, Vol: {volatility:.4%}")
    
    # Test 3: Regime-aware trading
    print("\n3. Testing Regime-Aware Trading...")
    regime_trader = RegimeAwareTrading(detector)
    
    # Create sample signal
    sample_signal = {
        "symbol": "US30Z25.sim",
        "direction": "BUY",
        "strength": 0.75,
        "position_size": 1.0
    }
    
    # Adapt signal based on regime
    adapted_signal = regime_trader.adapt_signals(sample_signal, regime_pred)
    
    print(f"✓ Original Position Size: {sample_signal['position_size']}")
    print(f"✓ Adapted Position Size: {adapted_signal['position_size']}")
    print(f"✓ Market Regime: {adapted_signal['market_regime']}")
    print(f"✓ Aggressiveness: {adapted_signal['regime_parameters']['aggressiveness']}")
    
    print("\n🎯 Basic functionality tests completed successfully!")

def test_integration():
    print("\nTesting System Integration...")
    
    # Simple integration test
    class SimpleRegimeSystem:
        def __init__(self):
            self.regime_detector = MarketRegimeDetector()
            self.regime_trader = RegimeAwareTrading(self.regime_detector)
            self.regime_history = []
        
        def initialize(self):
            dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
            prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
            price_series = pd.Series(prices, index=dates)
            self.regime_detector.train_model(price_series)
            print("✓ System initialized")
        
        def generate_signals(self):
            regime_pred = self.get_current_regime()
            
            # Create sample signals
            signals = []
            for symbol in ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]:
                signal = {
                    "symbol": symbol,
                    "direction": "BUY" if np.random.random() > 0.5 else "SELL",
                    "strength": np.random.uniform(0.5, 0.9),
                    "position_size": 1.0
                }
                adapted_signal = self.regime_trader.adapt_signals(signal, regime_pred)
                signals.append(adapted_signal)
            
            return signals
        
        def get_current_regime(self):
            dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
            prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
            price_series = pd.Series(prices, index=dates)
            
            regime_pred = self.regime_detector.predict_regime(price_series)
            self.regime_history.append(regime_pred)
            return regime_pred
    
    # Test the simple system
    system = SimpleRegimeSystem()
    system.initialize()
    signals = system.generate_signals()
    
    print(f"✓ Generated {len(signals)} regime-enhanced signals")
    for signal in signals[:2]:  # Show first 2 signals
        print(f"  {signal['symbol']}: {signal['direction']} (Regime: {signal['market_regime']})")
    
    print("✓ Integration test completed successfully!")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_integration()
        
        print("\n" + "="*50)
        print("🎯 MARKET REGIME DETECTION SYSTEM READY!")
        print("="*50)
        print("\nSystem Features:")
        print("✓ Hidden Markov Model with 3 market regimes")
        print("✓ Real-time regime detection and analysis")
        print("✓ Regime-aware trading signal adaptation")
        print("✓ Automatic position sizing based on volatility")
        print("✓ Conservative (High Vol) / Moderate (Med Vol) / Aggressive (Low Vol) strategies")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please check if all required packages are installed:")
        print("pip install hmmlearn scikit-learn pandas numpy matplotlib")
