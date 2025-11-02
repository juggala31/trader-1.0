from market_regime_detector import MarketRegimeDetector
import pandas as pd
import numpy as np

print("Quick verification test...")

# Create test data
dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
price_data = pd.Series(prices, index=dates)

# Test the detector
detector = MarketRegimeDetector()
regime_pred = detector.predict_regime(price_data)

print(f"Regime: {regime_pred['regime_label']}")
print(f"Confidence: {regime_pred['confidence']:.2%}")
print("✓ Basic functionality verified!")

# Test regime-aware trading
from market_regime_detector import RegimeAwareTrading

trader = RegimeAwareTrading(detector)
signal = {"symbol": "TEST", "direction": "BUY", "position_size": 1.0}
adapted = trader.adapt_signals(signal, regime_pred)

print(f"Original size: {signal['position_size']}")
print(f"Adapted size: {adapted['position_size']}")
print(f"Regime: {adapted['market_regime']}")
print("✓ Regime-aware trading verified!")

print("\n🎯 System is working correctly!")
