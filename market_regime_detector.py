import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class MarketRegimeDetector:
    def __init__(self, n_regimes=3, lookback_period=252):
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_labels = ["High Volatility", "Medium Volatility", "Low Volatility"]
        self.is_trained = False
        
    def extract_features(self, price_data):
        """Extract features for regime detection"""
        try:
            returns = price_data.pct_change().dropna()
            
            # Basic features
            features = pd.DataFrame()
            features["returns"] = returns
            
            # Volatility features
            features["volatility_20"] = returns.rolling(20, min_periods=5).std()
            features["volatility_50"] = returns.rolling(50, min_periods=10).std()
            
            # Momentum features
            features["momentum_10"] = price_data / price_data.shift(10) - 1
            features["momentum_20"] = price_data / price_data.shift(20) - 1
            
            # Simple RSI calculation
            def simple_rsi(prices, period=14):
                delta = prices.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(period, min_periods=5).mean()
                avg_loss = loss.rolling(period, min_periods=5).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            features["rsi_14"] = simple_rsi(price_data, 14)
            
            # Market regime features
            features["volatility_ratio"] = features["volatility_20"] / features["volatility_50"]
            features["trend_strength"] = features["momentum_20"].abs()
            
            # Handle NaN values
            features = features.fillna(method="bfill").fillna(method="ffill").fillna(0)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return simple returns if complex features fail
            returns = price_data.pct_change().dropna()
            return pd.DataFrame({"returns": returns}).fillna(0)
    
    def train_model(self, price_data):
        """Train HMM model on historical data"""
        try:
            features = self.extract_features(price_data)
            
            # Ensure we have enough data
            if len(features) < 50:
                print("Insufficient data for training. Need at least 50 samples.")
                return None
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train HMM
            self.hmm_model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            self.hmm_model.fit(features_scaled)
            self.is_trained = True
            
            print(f"✓ HMM Model trained with {self.n_regimes} regimes")
            return self.hmm_model
            
        except Exception as e:
            print(f"Training error: {e}")
            self.is_trained = False
            return None
    
    def predict_regime(self, price_data):
        """Predict current market regime"""
        if not self.is_trained or self.hmm_model is None:
            print("Model not trained. Training now...")
            self.train_model(price_data)
        
        try:
            features = self.extract_features(price_data)
            
            # Use recent data for prediction
            recent_features = features.tail(min(self.lookback_period, len(features)))
            features_scaled = self.scaler.transform(recent_features)
            
            # Predict regime probabilities
            regime_probs = self.hmm_model.predict_proba(features_scaled)
            current_regime_prob = regime_probs[-1]
            current_regime = np.argmax(current_regime_prob)
            
            return {
                "regime": current_regime,
                "regime_label": self.regime_labels[current_regime],
                "probabilities": current_regime_prob,
                "confidence": np.max(current_regime_prob)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback if prediction fails
            return {
                "regime": 1,
                "regime_label": "Medium Volatility",
                "probabilities": [0.33, 0.34, 0.33],
                "confidence": 0.34
            }
    
    def get_regime_statistics(self, price_data):
        """Analyze characteristics of each regime"""
        if not self.is_trained:
            self.train_model(price_data)
            
        try:
            features = self.extract_features(price_data)
            features_scaled = self.scaler.transform(features)
            regimes = self.hmm_model.predict(features_scaled)
            
            returns = price_data.pct_change().dropna()
            
            stats = {}
            for regime in range(self.n_regimes):
                # Ensure we have matching lengths
                if len(returns) >= len(regimes):
                    regime_mask = regimes == regime
                    if len(regime_mask) <= len(returns):
                        regime_returns = returns.iloc[:len(regime_mask)][regime_mask]
                    else:
                        regime_returns = returns
                else:
                    regime_returns = returns
                
                if len(regime_returns) > 0:
                    stats[regime] = {
                        "mean_return": regime_returns.mean(),
                        "volatility": regime_returns.std(),
                        "sharpe_ratio": regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                        "count": len(regime_returns),
                        "positive_ratio": (regime_returns > 0).mean()
                    }
                else:
                    stats[regime] = {
                        "mean_return": 0,
                        "volatility": 0,
                        "sharpe_ratio": 0,
                        "count": 0,
                        "positive_ratio": 0
                    }
            
            return stats
            
        except Exception as e:
            print(f"Statistics error: {e}")
            return {}

class RegimeAwareTrading:
    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        
    def get_regime_specific_parameters(self, regime_prediction):
        """Get trading parameters based on current regime"""
        regime = regime_prediction["regime"]
        
        if regime == 0:  # High volatility
            return {
                "position_size_multiplier": 0.5,
                "stop_loss_multiplier": 2.0,
                "take_profit_multiplier": 1.5,
                "max_drawdown_limit": 0.02,
                "aggressiveness": "conservative"
            }
        elif regime == 1:  # Medium volatility
            return {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 2.0,
                "max_drawdown_limit": 0.03,
                "aggressiveness": "moderate"
            }
        else:  # Low volatility
            return {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 2.5,
                "max_drawdown_limit": 0.04,
                "aggressiveness": "aggressive"
            }
    
    def high_volatility_strategy(self, signals, regime_params):
        """Conservative strategy for high volatility periods"""
        adapted_signals = signals.copy() if hasattr(signals, 'copy') else dict(signals)
        adapted_signals["position_size"] = adapted_signals.get("position_size", 1.0) * regime_params["position_size_multiplier"]
        return adapted_signals
    
    def medium_volatility_strategy(self, signals, regime_params):
        """Moderate strategy for medium volatility"""
        return signals
    
    def low_volatility_strategy(self, signals, regime_params):
        """Aggressive strategy for low volatility periods"""
        adapted_signals = signals.copy() if hasattr(signals, 'copy') else dict(signals)
        adapted_signals["position_size"] = adapted_signals.get("position_size", 1.0) * regime_params["position_size_multiplier"]
        return adapted_signals
    
    def adapt_signals(self, signals, regime_prediction):
        """Adapt trading signals based on market regime"""
        if signals is None:
            return None
            
        regime_params = self.get_regime_specific_parameters(regime_prediction)
        
        # Create a copy of signals
        adapted_signals = signals.copy() if hasattr(signals, 'copy') else dict(signals)
        
        # Apply regime-specific strategy
        regime = regime_prediction["regime"]
        if regime == 0:
            adapted_signals = self.high_volatility_strategy(adapted_signals, regime_params)
        elif regime == 1:
            adapted_signals = self.medium_volatility_strategy(adapted_signals, regime_params)
        else:
            adapted_signals = self.low_volatility_strategy(adapted_signals, regime_params)
        
        # Add regime information
        adapted_signals["market_regime"] = regime_prediction["regime_label"]
        adapted_signals["regime_confidence"] = regime_prediction["confidence"]
        adapted_signals["regime_parameters"] = regime_params
        
        return adapted_signals
