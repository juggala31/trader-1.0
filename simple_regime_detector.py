import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class SimpleMarketRegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.scaler = StandardScaler()
        
    def train_model(self, price_data):
        # Simple feature extraction - just returns
        returns = price_data.pct_change().dropna().values.reshape(-1, 1)
        
        # Scale returns
        returns_scaled = self.scaler.fit_transform(returns)
        
        # Train simple HMM
        self.hmm_model = GaussianHMM(n_components=self.n_regimes, n_iter=100)
        self.hmm_model.fit(returns_scaled)
        print("Simple HMM model trained successfully")
        
    def predict_regime(self, price_data):
        if self.hmm_model is None:
            self.train_model(price_data)
            
        returns = price_data.pct_change().dropna().values.reshape(-1, 1)
        returns_scaled = self.scaler.transform(returns)
        
        # Predict last regime
        regime = self.hmm_model.predict(returns_scaled)[-1]
        return {"regime": regime, "regime_label": f"Regime {regime}"}

# Test function
def test_simple():
    print("Testing simple market regime detection...")
    
    # Create simple test data
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    price_series = pd.Series(prices, index=dates)
    
    detector = SimpleMarketRegimeDetector()
    regime_pred = detector.predict_regime(price_series)
    
    print(f"Detected regime: {regime_pred['regime_label']}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_simple()
