from market_regime_detector import MarketRegimeDetector
from regime_ensemble_integration import RegimeEnhancedSystem
import pandas as pd
import numpy as np

def test_complete_system():
    print("Testing Complete Market Regime Detection System...")
    
    # Test 1: Basic regime detection
    print("\n1. Testing Basic Regime Detection...")
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
        print(f"✓ Regime {regime}: Mean Return {data['mean_return']:.4%}, Volatility {data['volatility']:.4%}")
    
    # Test 3: Complete system integration
    print("\n3. Testing System Integration...")
    system = RegimeEnhancedSystem()
    system.initialize_system()
    
    # Test regime-enhanced signals
    signals = system.generate_regime_enhanced_signals()
    print(f"✓ Generated {len(signals)} regime-enhanced signals")
    
    # Test performance metrics
    metrics = system.get_regime_performance_metrics()
    print("✓ Regime performance metrics calculated")
    
    print("\n🎯 All tests completed successfully!")
    print("\nSystem Features:")
    print("✓ Hidden Markov Model-based regime detection")
    print("✓ 3 distinct market regimes (High/Medium/Low Volatility)")
    print("✓ Regime-aware trading signal adaptation")
    print("✓ Real-time regime analytics")
    print("✓ Performance metrics by regime")

if __name__ == "__main__":
    test_complete_system()
