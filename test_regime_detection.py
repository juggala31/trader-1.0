from market_regime_detector import MarketRegimeDetector
import pandas as pd
import numpy as np

def test_regime_detection():
    print("Testing Market Regime Detection System...")
    
    # Create sample data for testing
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    price_series = pd.Series(prices, index=dates)
    
    # Test regime detection
    detector = MarketRegimeDetector()
    detector.train_model(price_series)
    regime_pred = detector.predict_regime(price_series)
    
    print(f"Market Regime: {regime_pred['regime_label']}")
    print(f"Confidence: {regime_pred['confidence']:.2%}")
    print(f"Probabilities: {regime_pred['probabilities']}")
    
    print("Market regime detection test completed successfully!")

if __name__ == "__main__":
    test_regime_detection()
