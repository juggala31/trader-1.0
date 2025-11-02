import numpy as np
import pandas as pd
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class OptimizedTradingQLearning:
    def __init__(self, learning_rate=0.2, discount_factor=0.95, exploration_rate=0.4):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        self.actions = [
            "BUY_AGGRESSIVE", "BUY_MODERATE", "BUY_CONSERVATIVE",
            "HOLD", 
            "SELL_AGGRESSIVE", "SELL_MODERATE", "SELL_CONSERVATIVE"
        ]
        
    def get_state(self, market_data, current_position=None):
        """Convert market data into a discrete state"""
        regime = market_data.get("regime", 1)
        trend = market_data.get("trend_strength", 0.5)
        volatility = market_data.get("volatility_level", 0.5)
        rsi = market_data.get("rsi", 50) / 100  # Normalize
        momentum = market_data.get("momentum", 0)
        
        # Discretize values
        state = (
            f"regime_{regime}",
            f"trend_{self._discretize(trend, 3)}",
            f"vol_{self._discretize(volatility, 3)}",
            f"rsi_{self._discretize(rsi, 3)}",
            f"mom_{'up' if momentum > 0 else 'down'}",
            f"pos_{'long' if current_position and current_position > 0 else 'short' if current_position and current_position < 0 else 'neutral'}"
        )
        
        return "_".join(state)
    
    def _discretize(self, value, bins):
        if value is None:
            return 1
        value = max(0, min(1, value))
        return min(bins - 1, int(value * bins))
    
    def choose_action(self, state, force_exploit=False):
        """Choose action using epsilon-greedy policy"""
        if not force_exploit and random.random() < self.exploration_rate:
            action = random.choice(self.actions)
        else:
            if state in self.q_table and self.q_table[state]:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = "HOLD"
        
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def calculate_reward(self, trade_result, market_data):
        """Calculate reward based on trade outcome"""
        if trade_result is None:
            return -0.05
        
        profit = trade_result.get("profit", 0)
        duration = trade_result.get("duration", 1)
        risk_taken = trade_result.get("risk", 0.01)
        
        if risk_taken > 0:
            risk_adjusted_return = profit / (risk_taken * max(1, duration))
        else:
            risk_adjusted_return = profit
        
        # Enhanced reward with regime consideration
        regime_multiplier = {
            0: 1.3,  # Higher reward for high volatility success
            1: 1.0,
            2: 0.8   # Lower reward for easy low volatility trades
        }.get(market_data.get("regime", 1), 1.0)
        
        reward = risk_adjusted_return * regime_multiplier
        
        return max(-5, min(5, reward))
    
    def get_performance_metrics(self):
        """Calculate RL performance metrics"""
        if not self.reward_history:
            return {"total_reward": 0, "avg_reward": 0, "success_rate": 0}
        
        rewards = np.array(self.reward_history)
        total_reward = rewards.sum()
        avg_reward = rewards.mean()
        success_rate = (rewards > 0).mean()
        
        return {
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "unique_states": len(set(self.state_history)) if self.state_history else 0
        }
    
    def decay_exploration(self, decay_rate=0.99):
        """Gradually reduce exploration rate"""
        self.exploration_rate = max(0.05, self.exploration_rate * decay_rate)

class OptimizedRLTradingSystem:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['BTCX25.sim', 'US30Z25.sim', 'XAUZ25.sim', 'US100Z25.sim', 'US500Z25.sim', 'USOILZ25.sim']
        from market_regime_detector import MarketRegimeDetector
        self.regime_detector = MarketRegimeDetector()
        self.rl_agent = OptimizedTradingQLearning()
        self.current_positions = {symbol: 0 for symbol in self.symbols}
        
    def initialize_system(self):
        """Initialize the RL-enhanced trading system"""
        historical_data = self._generate_sample_data(1000)
        self.regime_detector.train_model(historical_data)
        
        # Train RL agent on historical data
        training_data = self._prepare_training_data(historical_data)
        self._train_rl_agent(training_data, episodes=80)
        
        print("✓ Optimized RL trading system initialized")
        return True
    
    def _generate_sample_data(self, periods):
        """Generate sample market data with realistic trends"""
        dates = pd.date_range(start="2020-01-01", periods=periods, freq="D")
        # Create more realistic price series with some trends
        base_trend = 0.0002  # Small upward bias
        prices = [100]
        for i in range(1, periods):
            # Add trend + random noise
            change = base_trend + np.random.normal(0, 0.005)
            prices.append(prices[-1] * (1 + change))
        return pd.Series(prices, index=dates)
    
    def _prepare_training_data(self, price_data):
        """Prepare market data for RL training"""
        training_data = []
        
        for i in range(20, len(price_data) - 1):  # Use 20-day windows
            window_data = price_data.iloc[i-20:i]
            
            returns = window_data.pct_change().dropna()
            trend = (window_data.iloc[-1] / window_data.iloc[0] - 1) if len(window_data) > 1 else 0
            volatility = returns.std() if len(returns) > 1 else 0.01
            
            # Simple RSI calculation
            if len(window_data) >= 15:
                gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
                losses = -returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.001
                rsi = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50
            else:
                rsi = 50
            
            data_point = {
                "price": window_data.iloc[-1],
                "trend_strength": abs(trend),
                "volatility_level": min(1.0, volatility * 10),  # Scale volatility
                "rsi": rsi,
                "momentum": trend,
                "regime": 1  # Default for training
            }
            
            training_data.append(data_point)
        
        return training_data
    
    def _train_rl_agent(self, training_data, episodes=50):
        """Train RL agent on historical data"""
        print(f"Training optimized RL agent on {episodes} episodes...")
        
        for episode in range(episodes):
            episode_reward = 0
            
            for i in range(len(training_data) - 1):
                current_data = training_data[i]
                next_data = training_data[i + 1]
                
                state = self.rl_agent.get_state(current_data)
                action = self.rl_agent.choose_action(state)
                
                # Simulate trade result
                trade_result = self._simulate_trade_for_training(action, current_data, next_data)
                
                next_state = self.rl_agent.get_state(next_data)
                reward = self.rl_agent.calculate_reward(trade_result, next_data)
                
                self.rl_agent.update_q_value(state, action, reward, next_state)
                episode_reward += reward
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")
            
            self.rl_agent.decay_exploration()
    
    def _simulate_trade_for_training(self, action, current_data, next_data):
        """Simulate trade result for training"""
        price_change = (next_data.get("price", 100) - current_data.get("price", 100)) / current_data.get("price", 100)
        
        # Action effectiveness
        action_multiplier = {
            "AGGRESSIVE": 1.5,
            "MODERATE": 1.0,
            "CONSERVATIVE": 0.6
        }
        
        multiplier = 1.0
        for level in action_multiplier:
            if level in action:
                multiplier = action_multiplier[level]
                break
        
        if "BUY" in action and price_change > 0:
            profit = price_change * 100 * multiplier
        elif "SELL" in action and price_change < 0:
            profit = abs(price_change) * 100 * multiplier
        else:
            profit = -abs(price_change) * 50  # Smaller penalty for wrong direction
        
        return {
            "profit": profit,
            "duration": 1,
            "risk": 0.01 * multiplier
        }
    
    def generate_trading_signals(self):
        """Generate RL-enhanced trading signals"""
        signals = []
        
        for symbol in self.symbols:
            market_data = self._get_current_market_data(symbol)
            
            state = self.rl_agent.get_state(market_data, self.current_positions.get(symbol, 0))
            action = self.rl_agent.choose_action(state)
            
            signal = self._action_to_signal(action, market_data, symbol, state)
            signals.append(signal)
        
        return signals
    
    def _get_current_market_data(self, symbol):
        """Get current market data for RL decision"""
        price_data = self._generate_sample_data(50)  # Last 50 days
        regime_pred = self.regime_detector.predict_regime(price_data)
        
        returns = price_data.pct_change().dropna()
        trend = (price_data.iloc[-1] / price_data.iloc[0] - 1) if len(price_data) > 1 else 0
        volatility = returns.std() if len(returns) > 1 else 0.01
        
        # Simple RSI
        if len(price_data) >= 15:
            gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            losses = -returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.001
            rsi = 100 - (100 / (1 + gains/losses)) if losses != 0 else 50
        else:
            rsi = 50
        
        return {
            "price_data": price_data,
            "price": price_data.iloc[-1],
            "regime": regime_pred["regime"],
            "trend_strength": abs(trend),
            "volatility_level": min(1.0, volatility * 10),
            "rsi": rsi,
            "momentum": trend,
            "symbol": symbol
        }
    
    def _action_to_signal(self, action, market_data, symbol, state):
        """Convert RL action to trading signal"""
        signal_map = {
            "BUY_AGGRESSIVE": {"action": "BUY", "size_multiplier": 1.5, "aggressiveness": "high"},
            "BUY_MODERATE": {"action": "BUY", "size_multiplier": 1.0, "aggressiveness": "medium"},
            "BUY_CONSERVATIVE": {"action": "BUY", "size_multiplier": 0.6, "aggressiveness": "low"},
            "HOLD": {"action": "HOLD", "size_multiplier": 0, "aggressiveness": "none"},
            "SELL_AGGRESSIVE": {"action": "SELL", "size_multiplier": 1.5, "aggressiveness": "high"},
            "SELL_MODERATE": {"action": "SELL", "size_multiplier": 1.0, "aggressiveness": "medium"},
            "SELL_CONSERVATIVE": {"action": "SELL", "size_multiplier": 0.6, "aggressiveness": "low"}
        }
        
        signal = signal_map.get(action, signal_map["HOLD"])
        signal.update({
            "symbol": symbol,
            "rl_state": state,
            "rl_action": action,
            "timestamp": pd.Timestamp.now()
        })
        
        regime_pred = self.regime_detector.predict_regime(market_data.get("price_data", []))
        signal["market_regime"] = regime_pred["regime_label"]
        signal["regime_confidence"] = regime_pred["confidence"]
        
        return signal
    
    def execute_trade(self, signal, simulated=True):
        """Execute trade and update RL learning"""
        if simulated:
            trade_result = self._simulate_trade(signal)
        else:
            trade_result = {"profit": 0, "duration": 1, "risk": 0.01}
        
        # Update RL agent
        next_market_data = self._get_current_market_data(signal["symbol"])
        next_state = self.rl_agent.get_state(next_market_data, self.current_positions.get(signal["symbol"], 0))
        reward = self.rl_agent.calculate_reward(trade_result, next_market_data)
        
        self.rl_agent.update_q_value(signal["rl_state"], signal["rl_action"], reward, next_state)
        
        # Update position
        if signal["action"] == "BUY":
            self.current_positions[signal["symbol"]] = 1
        elif signal["action"] == "SELL":
            self.current_positions[signal["symbol"]] = -1
        
        return trade_result
    
    def _simulate_trade(self, signal):
        """Simulate trade outcome"""
        # More realistic simulation
        base_profit = np.random.normal(0.001, 0.006)
        
        # Action effectiveness
        action_multiplier = signal.get("size_multiplier", 1.0)
        
        if signal["action"] == "BUY":
            profit = base_profit * action_multiplier
        elif signal["action"] == "SELL":
            profit = -base_profit * action_multiplier
        else:
            profit = 0
        
        return {
            "profit": profit * 10000,
            "duration": max(1, int(np.random.exponential(1.5))),
            "risk": 0.01 * action_multiplier
        }
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        signals = self.generate_trading_signals()
        
        total_profit = 0
        for signal in signals:
            trade_result = self.execute_trade(signal, simulated=True)
            total_profit += trade_result["profit"]
            
            print(f"{signal['symbol']}: {signal['rl_action']} -> ${trade_result['profit']:.2f}")
        
        # Update exploration
        self.rl_agent.decay_exploration()
        
        return total_profit
    
    def get_system_metrics(self):
        """Get comprehensive system performance metrics"""
        rl_metrics = self.rl_agent.get_performance_metrics()
        
        return {
            "rl_learning": rl_metrics,
            "current_positions": self.current_positions,
            "total_trades": len(self.rl_agent.action_history)
        }
    
    def run_optimized_demo(self):
        """Run demo with optimized RL"""
        print("🚀 OPTIMIZED RL Trading System Demo")
        print("=" * 50)
        
        self.initialize_system()
        
        # Run multiple cycles to show learning
        total_demo_profit = 0
        for cycle in range(6):
            print(f"\\n--- Trading Cycle {cycle + 1} ---")
            cycle_profit = self.run_trading_cycle()
            total_demo_profit += cycle_profit
            
            metrics = self.get_system_metrics()
            print(f"Cycle Profit: ${cycle_profit:.2f}")
            print(f"Success Rate: {metrics['rl_learning']['success_rate']:.2%}")
            print(f"Exploration Rate: {metrics['rl_learning']['exploration_rate']:.3f}")
        
        final_metrics = self.get_system_metrics()
        print("\\n" + "=" * 50)
        print("🎯 OPTIMIZED SYSTEM RESULTS:")
        print("=" * 50)
        print(f"Total Demo Profit: ${total_demo_profit:.2f}")
        print(f"Total Reward: {final_metrics['rl_learning']['total_reward']:.2f}")
        print(f"Average Reward: {final_metrics['rl_learning']['avg_reward']:.3f}")
        print(f"Success Rate: {final_metrics['rl_learning']['success_rate']:.2%}")
        print(f"Unique States: {final_metrics['rl_learning']['unique_states']}")
        print(f"Final Exploration: {final_metrics['rl_learning']['exploration_rate']:.3f}")

def demo_optimized_rl():
    """Demo the optimized RL system"""
    system = OptimizedRLTradingSystem()
    system.run_optimized_demo()

if __name__ == "__main__":
    demo_optimized_rl()

