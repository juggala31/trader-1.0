# Performance Optimizer - Real-time Strategy Optimization
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize

class PerformanceOptimizer:
    def __init__(self, ftmo_logger):
        self.ftmo_logger = ftmo_logger
        self.optimization_history = []
        self.current_parameters = self._get_default_parameters()
        self.last_optimization = None
        self.optimization_interval = timedelta(hours=6)  # Optimize every 6 hours
        
    def _get_default_parameters(self):
        """Get default strategy parameters"""
        return {
            'ema_fast_period': 20,
            'ema_slow_period': 50,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'atr_multiplier_sl': 1.2,
            'atr_multiplier_tp': 2.0,
            'confidence_threshold': 0.6
        }
        
    def should_optimize(self):
        """Check if it's time to optimize parameters"""
        if self.last_optimization is None:
            return True
            
        time_since_last = datetime.now() - self.last_optimization
        return time_since_last >= self.optimization_interval
        
    def optimize_parameters(self):
        """Optimize strategy parameters based on recent performance"""
        if not self.should_optimize():
            return self.current_parameters
            
        logging.info("Starting parameter optimization...")
        
        try:
            # Get recent performance data
            recent_trades = self._get_recent_trades(100)  # Last 100 trades
            if len(recent_trades) < 20:
                logging.warning("Not enough trades for optimization")
                return self.current_parameters
                
            # Prepare optimization data
            X, y = self._prepare_optimization_data(recent_trades)
            
            # Define optimization objective
            def objective(params):
                return -self._evaluate_parameters(params, X, y)  # Negative for minimization
                
            # Initial parameters
            x0 = list(self.current_parameters.values())
            
            # Parameter bounds
            bounds = [
                (10, 30),    # ema_fast_period
                (40, 60),    # ema_slow_period  
                (25, 35),    # rsi_oversold
                (65, 75),    # rsi_overbought
                (1.0, 1.5),  # atr_sl_multiplier
                (1.5, 3.0),  # atr_tp_multiplier
                (0.5, 0.8)   # confidence_threshold
            ]
            
            # Run optimization
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimized_params = self._vector_to_parameters(result.x)
                
                # Validate improvement
                old_score = self._evaluate_parameters(x0, X, y)
                new_score = self._evaluate_parameters(result.x, X, y)
                
                if new_score > old_score * 1.05:  # 5% improvement required
                    self.current_parameters = optimized_params
                    self.last_optimization = datetime.now()
                    
                    logging.info(f"Parameters optimized: Score improved from {old_score:.3f} to {new_score:.3f}")
                    
                    # Record optimization
                    self.optimization_history.append({
                        'timestamp': datetime.now(),
                        'old_score': old_score,
                        'new_score': new_score,
                        'parameters': optimized_params
                    })
                    
                else:
                    logging.info("Optimization didn't provide significant improvement")
                    
            else:
                logging.warning("Optimization failed")
                
        except Exception as e:
            logging.error(f"Optimization error: {e}")
            
        return self.current_parameters
        
    def _prepare_optimization_data(self, trades):
        """Prepare data for optimization"""
        # This would use actual market conditions during trades
        # For now, return placeholder data
        X = np.random.rand(len(trades), 5)  # 5 features
        y = np.array([1 if trade.get('profit', 0) > 0 else 0 for trade in trades])
        return X, y
        
    def _evaluate_parameters(self, params, X, y):
        """Evaluate parameters using objective function"""
        # Simplified evaluation - would use actual backtesting
        win_rate = np.mean(y) if len(y) > 0 else 0.5
        avg_profit = np.mean([trade.get('profit', 0) for trade in self._get_recent_trades(20)])
        
        # Combine metrics
        score = win_rate * 0.7 + (avg_profit / 100) * 0.3  # Normalize profit
        return max(0, score)  # Ensure non-negative
        
    def _vector_to_parameters(self, vector):
        """Convert optimization vector to parameter dictionary"""
        return {
            'ema_fast_period': int(vector[0]),
            'ema_slow_period': int(vector[1]),
            'rsi_oversold': int(vector[2]),
            'rsi_overbought': int(vector[3]),
            'atr_multiplier_sl': round(vector[4], 2),
            'atr_multiplier_tp': round(vector[5], 2),
            'confidence_threshold': round(vector[6], 2)
        }
        
    def _get_recent_trades(self, count):
        """Get recent trades"""
        return self.ftmo_logger.trade_history[-count:] if self.ftmo_logger.trade_history else []
        
    def get_optimization_status(self):
        """Get optimization status report"""
        return {
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'current_parameters': self.current_parameters,
            'optimization_count': len(self.optimization_history),
            'next_optimization': (self.last_optimization + self.optimization_interval).isoformat() if self.last_optimization else None
        }
