import pandas as pd
import numpy as np
from reinforcement_learning import RLEnhancedTrading
from market_regime_detector import MarketRegimeDetector

class RLEnhancedTradingSystem:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['BTCX25.sim', 'US30Z25.sim', 'XAUZ25.sim', 'US100Z25.sim', 'US500Z25.sim', 'USOILZ25.sim']
        self.regime_detector = MarketRegimeDetector()
        self.rl_trader = RLEnhancedTrading(self.regime_detector)
        self.current_positions = {symbol: 0 for symbol in self.symbols}
        self.performance_history = []
        
    def initialize_system(self):
        """Initialize the RL-enhanced trading system"""
        # Initialize regime detection
        historical_data = self._generate_sample_data(1000)
        self.regime_detector.train_model(historical_data)
        
        # Train RL agent on historical data
        training_data = self._prepare_training_data(historical_data)
        self.rl_trader.train_on_historical_data(training_data, episodes=50)
        
        print("✓ RL-enhanced trading system initialized")
        return True
    
    def _generate_sample_data(self, periods):
        """Generate sample market data"""
        dates = pd.date_range(start="2020-01-01", periods=periods, freq="D")
        prices = 100 + np.cumsum(np.random.randn(periods) * 0.5)
        return pd.Series(prices, index=dates)
    
    def _prepare_training_data(self, price_data):
        """Prepare market data for RL training"""
        training_data = []
        
        for i in range(len(price_data) - 20):  # Use 20-day windows
            window_data = price_data.iloc[i:i+20]
            
            # Calculate features for this window
            returns = window_data.pct_change().dropna()
            trend = (window_data.iloc[-1] / window_data.iloc[0] - 1) if len(window_data) > 1 else 0
            volatility = returns.std() if len(returns) > 1 else 0.01
            rsi = self._calculate_simple_rsi(window_data)
            momentum = trend
            
            data_point = {
                "price": window_data.iloc[-1],
                "trend_strength": abs(trend),
                "volatility_level": volatility,
                "rsi": rsi,
                "momentum": momentum,
                "regime": 1  # Default to medium volatility for training
            }
            
            training_data.append(data_point)
        
        return training_data
    
    def _calculate_simple_rsi(self, prices, period=14):
        """Calculate simple RSI"""
        if len(prices) < period + 1:
            return 50
        
        returns = prices.pct_change().dropna()
        gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        losses = -returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.001
        
        if losses == 0:
            return 100
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_trading_signals(self):
        """Generate RL-enhanced trading signals"""
        signals = []
        
        for symbol in self.symbols:
            # Get current market data (simulated)
            market_data = self._get_current_market_data(symbol)
            
            # Generate RL signal
            signal, state, action = self.rl_trader.generate_rl_signals(
                market_data, self.current_positions.get(symbol, 0)
            )
            
            signal["symbol"] = symbol
            signal["rl_state"] = state
            signal["rl_action"] = action
            
            signals.append(signal)
        
        return signals
    
    def _get_current_market_data(self, symbol):
        """Get current market data for RL decision"""
        # Simulate market data - in real system, this would come from MT5
        price_data = self._generate_sample_data(50)  # Last 50 days
        regime_pred = self.regime_detector.predict_regime(price_data)
        
        returns = price_data.pct_change().dropna()
        trend = (price_data.iloc[-1] / price_data.iloc[0] - 1) if len(price_data) > 1 else 0
        volatility = returns.std() if len(returns) > 1 else 0.01
        rsi = self._calculate_simple_rsi(price_data)
        
        return {
            "price_data": price_data,
            "price": price_data.iloc[-1],
            "regime": regime_pred["regime"],
            "trend_strength": abs(trend),
            "volatility_level": volatility,
            "rsi": rsi,
            "momentum": trend,
            "symbol": symbol
        }
    
    def execute_trade(self, signal, simulated=True):
        """Execute trade and update RL learning"""
        if simulated:
            # Simulate trade execution
            trade_result = self._simulate_trade(signal)
        else:
            # Real trade execution would go here
            trade_result = {"profit": 0, "duration": 1, "risk": 0.01}
        
        # Update RL agent
        next_market_data = self._get_current_market_data(signal["symbol"])
        self.rl_trader.update_learning(
            trade_result,
            signal["rl_state"],
            signal["rl_action"],
            next_market_data,
            self.current_positions.get(signal["symbol"], 0)
        )
        
        # Update position
        if signal["action"] == "BUY":
            self.current_positions[signal["symbol"]] = 1
        elif signal["action"] == "SELL":
            self.current_positions[signal["symbol"]] = -1
        elif signal["action"] == "HOLD":
            pass  # Maintain current position
        
        return trade_result
    
    def _simulate_trade(self, signal):
        """Simulate trade outcome for demonstration"""
        # Simple simulation based on action aggressiveness
        size_multiplier = signal.get("size_multiplier", 1.0)
        base_profit = np.random.normal(0.001, 0.005)  # Small random profit/loss
        
        # Adjust based on action type and aggressiveness
        if signal["action"] == "BUY":
            profit = base_profit * size_multiplier
        elif signal["action"] == "SELL":
            profit = -base_profit * size_multiplier
        else:  # HOLD
            profit = 0
        
        return {
            "profit": profit * 10000,  # Scale for demonstration
            "duration": 1,
            "risk": 0.01 * size_multiplier
        }
    
    def get_system_metrics(self):
        """Get comprehensive system performance metrics"""
        rl_metrics = self.rl_trader.get_learning_metrics()
        regime_metrics = self._get_regime_metrics()
        
        return {
            "rl_learning": rl_metrics,
            "market_regimes": regime_metrics,
            "current_positions": self.current_positions,
            "total_trades": len(self.rl_trader.trade_history)
        }
    
    def _get_regime_metrics(self):
        """Get market regime metrics"""
        price_data = self._generate_sample_data(200)
        regime_pred = self.regime_detector.predict_regime(price_data)
        
        return {
            "current_regime": regime_pred["regime_label"],
            "confidence": regime_pred["confidence"],
            "regime_history": len(self.regime_detector.regime_history) if hasattr(self.regime_detector, 'regime_history') else 0
        }
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        print("\\n=== RL Trading Cycle ===")
        
        # Generate signals
        signals = self.generate_trading_signals()
        
        # Execute trades and learn
        for signal in signals:
            print(f"Symbol: {signal['symbol']}")
            print(f"RL Action: {signal['rl_action']}")
            print(f"Market Regime: {signal.get('market_regime', 'Unknown')}")
            
            trade_result = self.execute_trade(signal, simulated=True)
            print(f"Trade Result: ${trade_result['profit']:.2f}")
            print("-" * 40)
        
        # Show learning progress
        metrics = self.get_system_metrics()
        print(f"Total Reward: {metrics['rl_learning']['total_reward']:.2f}")
        print(f"Success Rate: {metrics['rl_learning']['success_rate']:.2%}")
        print(f"Exploration Rate: {metrics['rl_learning']['exploration_rate']:.3f}")

def demo_rl_system():
    """Demonstrate the RL-enhanced trading system"""
    print("🚀 FTMO Trading System - Reinforcement Learning Demo")
    print("=" * 50)
    
    system = RLEnhancedTradingSystem()
    system.initialize_system()
    
    # Run multiple trading cycles to demonstrate learning
    for cycle in range(5):
        print(f"\\n--- Trading Cycle {cycle + 1} ---")
        system.run_trading_cycle()
    
    # Final metrics
    metrics = system.get_system_metrics()
    print("\\n" + "=" * 50)
    print("🎯 FINAL SYSTEM METRICS:")
    print("=" * 50)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total Reward: {metrics['rl_learning']['total_reward']:.2f}")
    print(f"Average Reward: {metrics['rl_learning']['avg_reward']:.2f}")
    print(f"Success Rate: {metrics['rl_learning']['success_rate']:.2%}")
    print(f"Current Regime: {metrics['market_regimes']['current_regime']}")
    print(f"Regime Confidence: {metrics['market_regimes']['confidence']:.2%}")

if __name__ == "__main__":
    demo_rl_system()

