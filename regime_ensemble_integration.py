import pandas as pd
import numpy as np
from market_regime_detector import MarketRegimeDetector, RegimeAwareTrading

class RegimeEnhancedSystem:
    def __init__(self, symbols=None):
        self.regime_detector = MarketRegimeDetector(n_regimes=3)
        self.regime_trader = RegimeAwareTrading(self.regime_detector)
        self.symbols = symbols or ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]
        self.regime_history = []
        
    def initialize_system(self, historical_data=None):
        """Initialize regime detection with historical data"""
        if historical_data is None:
            historical_data = self.generate_sample_data()
        
        self.regime_detector.train_model(historical_data)
        print("Market regime detection system initialized")
        return True
    
    def generate_sample_data(self, periods=1000):
        """Generate sample price data for testing"""
        dates = pd.date_range(start="2020-01-01", periods=periods, freq="D")
        prices = 100 + np.cumsum(np.random.randn(periods) * 0.5)
        return pd.Series(prices, index=dates)
    
    def get_current_regime_analysis(self):
        """Get current market regime analysis"""
        price_data = self.generate_sample_data(200)  # Last 200 days
        regime_prediction = self.regime_detector.predict_regime(price_data)
        
        # Store regime history
        self.regime_history.append({
            "timestamp": pd.Timestamp.now(),
            "regime": regime_prediction["regime"],
            "regime_label": regime_prediction["regime_label"],
            "confidence": regime_prediction["confidence"]
        })
        
        return regime_prediction
    
    def generate_sample_signals(self):
        """Generate sample trading signals for testing"""
        signals = []
        for symbol in self.symbols:
            signal = {
                "symbol": symbol,
                "direction": "BUY" if np.random.random() > 0.5 else "SELL",
                "strength": np.random.uniform(0.5, 0.9),
                "position_size": 1.0,
                "timestamp": pd.Timestamp.now()
            }
            signals.append(signal)
        return signals
    
    def generate_regime_enhanced_signals(self):
        """Generate trading signals enhanced with regime awareness"""
        # Get base signals (in real system, this would come from your ensemble)
        base_signals = self.generate_sample_signals()
        
        # Get current regime
        regime_analysis = self.get_current_regime_analysis()
        
        # Adapt signals based on regime
        enhanced_signals = []
        for signal in base_signals:
            adapted_signal = self.regime_trader.adapt_signals(signal, regime_analysis)
            enhanced_signals.append(adapted_signal)
        
        return enhanced_signals
    
    def get_regime_performance_metrics(self):
        """Analyze performance by market regime"""
        if len(self.regime_history) < 5:
            return "Insufficient regime history for analysis"
        
        regime_df = pd.DataFrame(self.regime_history)
        
        metrics = {}
        for regime in range(3):
            regime_data = regime_df[regime_df["regime"] == regime]
            if len(regime_data) > 0:
                metrics[regime] = {
                    "frequency": len(regime_data) / len(regime_df),
                    "avg_confidence": regime_data["confidence"].mean(),
                    "duration_hours": self.calculate_avg_regime_duration(regime)
                }
        
        return metrics
    
    def calculate_avg_regime_duration(self, regime):
        """Calculate average duration of each regime"""
        regime_df = pd.DataFrame(self.regime_history)
        if len(regime_df) < 2:
            return 0
            
        regime_changes = regime_df["regime"].diff().ne(0)
        regime_periods = regime_changes.cumsum()
        
        durations = []
        for period in regime_periods.unique():
            period_data = regime_df[regime_periods == period]
            if len(period_data) > 0 and period_data["regime"].iloc[0] == regime:
                duration = (period_data["timestamp"].iloc[-1] - period_data["timestamp"].iloc[0]).total_seconds() / 3600
                durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def run_demo(self):
        """Run a demo of the regime-enhanced system"""
        print("=== FTMO Trading System - Market Regime Demo ===")
        
        # Initialize system
        self.initialize_system()
        
        # Generate and display enhanced signals
        signals = self.generate_regime_enhanced_signals()
        
        print("\n=== Regime-Enhanced Trading Signals ===")
        for signal in signals:
            print(f"Symbol: {signal['symbol']}")
            print(f"Direction: {signal['direction']}")
            print(f"Market Regime: {signal['market_regime']}")
            print(f"Regime Confidence: {signal['regime_confidence']:.2%}")
            print(f"Aggressiveness: {signal['regime_parameters']['aggressiveness']}")
            print(f"Position Size Multiplier: {signal['regime_parameters']['position_size_multiplier']}")
            print("-" * 40)
        
        # Show performance metrics
        metrics = self.get_regime_performance_metrics()
        print("\n=== Regime Performance Metrics ===")
        if isinstance(metrics, str):
            print(metrics)
        else:
            for regime, data in metrics.items():
                print(f"Regime {regime}:")
                print(f"  Frequency: {data['frequency']:.2%}")
                print(f"  Avg Confidence: {data['avg_confidence']:.2%}")
                print(f"  Avg Duration: {data['duration_hours']:.1f} hours")
        
        print("\nDemo completed successfully!")
