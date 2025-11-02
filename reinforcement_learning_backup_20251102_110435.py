import numpy as np
import pandas as pd
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class TradingQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Define trading actions
        self.actions = [
            "BUY_AGGRESSIVE", "BUY_MODERATE", "BUY_CONSERVATIVE",
            "HOLD", 
            "SELL_AGGRESSIVE", "SELL_MODERATE", "SELL_CONSERVATIVE"
        ]
        
        # Define state features
        self.state_features = [
            "market_regime", "trend_strength", "volatility_level", 
            "rsi_signal", "momentum_direction", "position_status"
        ]
    
    def get_state(self, market_data, current_position=None):
        """Convert market data into a discrete state"""
        # Extract features from market data
        regime = market_data.get("regime", 1)  # 0=High, 1=Medium, 2=Low
        trend = self._get_trend_strength(market_data)
        volatility = self._get_volatility_level(market_data)
        rsi = self._get_rsi_signal(market_data)
        momentum = self._get_momentum_direction(market_data)
        position = self._get_position_status(current_position)
        
        # Discretize continuous values
        state = (
            f"regime_{regime}",
            f"trend_{self._discretize(trend, 3)}",
            f"vol_{self._discretize(volatility, 3)}",
            f"rsi_{self._discretize(rsi, 3)}",
            f"mom_{momentum}",
            f"pos_{position}"
        )
        
        return "_".join(state)
    
    def _discretize(self, value, bins):
        """Convert continuous value to discrete bin"""
        if value is None:
            return 1
        return min(bins - 1, max(0, int(value * bins)))
    
    def _get_trend_strength(self, market_data):
        return market_data.get("trend_strength", 0.5)
    
    def _get_volatility_level(self, market_data):
        return market_data.get("volatility_level", 0.5)
    
    def _get_rsi_signal(self, market_data):
        rsi = market_data.get("rsi", 50)
        return (rsi - 30) / 40  # Normalize to 0-1 range
    
    def _get_momentum_direction(self, market_data):
        momentum = market_data.get("momentum", 0)
        return "up" if momentum > 0 else "down"
    
    def _get_position_status(self, position):
        if position is None or position == 0:
            return "neutral"
        elif position > 0:
            return "long"
        else:
            return "short"
    
    def choose_action(self, state, force_exploit=False):
        """Choose action using epsilon-greedy policy"""
        if not force_exploit and random.random() < self.exploration_rate:
            # Exploration: random action
            action = random.choice(self.actions)
        else:
            # Exploitation: best known action
            if state in self.q_table and self.q_table[state]:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = "HOLD"  # Default action
        
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Store history for analysis
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def calculate_reward(self, trade_result, market_data):
        """Calculate reward based on trade outcome"""
        if trade_result is None:
            return -0.1  # Small penalty for no action
        
        profit = trade_result.get("profit", 0)
        duration = trade_result.get("duration", 1)
        risk_taken = trade_result.get("risk", 0.01)
        
        # Normalize profit by risk and duration
        if risk_taken > 0:
            risk_adjusted_return = profit / (risk_taken * max(1, duration))
        else:
            risk_adjusted_return = profit
        
        # Reward shaping based on market conditions
        regime_bonus = {
            0: 0.5,  # Conservative bonus in high volatility
            1: 1.0,  # Standard in medium volatility
            2: 1.5   # Aggressive bonus in low volatility
        }.get(market_data.get("regime", 1), 1.0)
        
        reward = risk_adjusted_return * regime_bonus
        
        # Clip reward to reasonable range
        return max(-10, min(10, reward))
    
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
            "exploration_rate": self.exploration_rate
        }
    
    def decay_exploration(self, decay_rate=0.995):
        """Gradually reduce exploration rate"""
        self.exploration_rate = max(0.01, self.exploration_rate * decay_rate)
    
    def save_model(self, filepath):
        """Save Q-table to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filepath):
        """Load Q-table from file"""
        import pickle
        with open(filepath, 'rb') as f:
            loaded_q = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), loaded_q)

class RLEnhancedTrading:
    def __init__(self, regime_detector=None):
        self.rl_agent = TradingQLearning()
        self.regime_detector = regime_detector
        self.trade_history = []
        
    def generate_rl_signals(self, market_data, current_positions=None):
        """Generate trading signals using RL agent"""
        # Get current state
        state = self.rl_agent.get_state(market_data, current_positions)
        
        # Choose action using RL
        action = self.rl_agent.choose_action(state)
        
        # Convert RL action to trading signal
        signal = self._action_to_signal(action, market_data)
        
        return signal, state, action
    
    def _action_to_signal(self, action, market_data):
        """Convert RL action to trading signal format"""
        signal_map = {
            "BUY_AGGRESSIVE": {"action": "BUY", "size_multiplier": 1.5, "aggressiveness": "high"},
            "BUY_MODERATE": {"action": "BUY", "size_multiplier": 1.0, "aggressiveness": "medium"},
            "BUY_CONSERVATIVE": {"action": "BUY", "size_multiplier": 0.5, "aggressiveness": "low"},
            "HOLD": {"action": "HOLD", "size_multiplier": 0, "aggressiveness": "none"},
            "SELL_AGGRESSIVE": {"action": "SELL", "size_multiplier": 1.5, "aggressiveness": "high"},
            "SELL_MODERATE": {"action": "SELL", "size_multiplier": 1.0, "aggressiveness": "medium"},
            "SELL_CONSERVATIVE": {"action": "SELL", "size_multiplier": 0.5, "aggressiveness": "low"}
        }
        
        base_signal = signal_map.get(action, signal_map["HOLD"])
        
        # Enhance with market regime information
        if self.regime_detector:
            regime_pred = self.regime_detector.predict_regime(market_data.get("price_data", []))
            base_signal["market_regime"] = regime_pred["regime_label"]
            base_signal["regime_confidence"] = regime_pred["confidence"]
        
        base_signal["rl_action"] = action
        base_signal["timestamp"] = pd.Timestamp.now()
        
        return base_signal
    
    def update_learning(self, trade_result, state, action, next_market_data, next_positions):
        """Update RL agent based on trade outcome"""
        next_state = self.rl_agent.get_state(next_market_data, next_positions)
        reward = self.rl_agent.calculate_reward(trade_result, next_market_data)
        
        self.rl_agent.update_q_value(state, action, reward, next_state)
        
        # Store trade history
        self.trade_history.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "timestamp": pd.Timestamp.now(),
            "trade_result": trade_result
        })
        
        # Gradually reduce exploration
        self.rl_agent.decay_exploration()
    
    def get_learning_metrics(self):
        """Get RL learning performance metrics"""
        return self.rl_agent.get_performance_metrics()
    
    def train_on_historical_data(self, historical_data, episodes=100):
        """Train RL agent on historical data"""
        print(f"Training RL agent on {episodes} episodes...")
        
        for episode in range(episodes):
            episode_reward = 0
            
            for i in range(len(historical_data) - 1):
                current_data = historical_data[i]
                next_data = historical_data[i + 1]
                
                # Simulate trading decision
                state = self.rl_agent.get_state(current_data)
                action = self.rl_agent.choose_action(state, force_exploit=False)
                
                # Simulate trade result (simplified)
                trade_result = self._simulate_trade_result(action, current_data, next_data)
                
                next_state = self.rl_agent.get_state(next_data)
                reward = self.rl_agent.calculate_reward(trade_result, next_data)
                
                self.rl_agent.update_q_value(state, action, reward, next_state)
                episode_reward += reward
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")
    
    def _simulate_trade_result(self, action, current_data, next_data):
        """Simulate trade result for training"""
        # Simplified simulation - in real system, this would use actual trading logic
        price_change = (next_data.get("price", 100) - current_data.get("price", 100)) / current_data.get("price", 100)
        
        if "BUY" in action and price_change > 0:
            profit = abs(price_change) * 100
        elif "SELL" in action and price_change < 0:
            profit = abs(price_change) * 100
        else:
            profit = -abs(price_change) * 50  # Penalty for wrong direction
        
        return {
            "profit": profit,
            "duration": 1,
            "risk": 0.01
        }
