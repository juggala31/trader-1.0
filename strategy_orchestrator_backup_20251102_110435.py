# Strategy Orchestrator - Manages XGBoost vs Fallback strategy switching
import zmq
import json
import logging
import time
from datetime import datetime
import threading

class StrategyOrchestrator:
    def __init__(self, config):
        self.config = config
        self.current_strategy = "xgboost"
        self.fallback_triggered = False
        self.performance_metrics = {
            'xgboost_win_rate': 0.0,
            'fallback_win_rate': 0.0,
            'last_switch_time': None,
            'switch_count': 0
        }
        
        # ZMQ setup for strategy communication
        self.context = zmq.Context()
        self.strategy_pub = self.context.socket(zmq.PUB)
        self.strategy_pub.bind("tcp://*:5556")
        
        self.performance_sub = self.context.socket(zmq.SUB)
        self.performance_sub.connect("tcp://localhost:5557")
        self.performance_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_strategies, daemon=True)
        self.monitor_thread.start()
        
        logging.info("Strategy Orchestrator initialized")
        
    def _monitor_strategies(self):
        """Continuous monitoring of strategy performance"""
        while True:
            try:
                self.evaluate_strategy_performance()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logging.error(f"Strategy monitoring error: {e}")
                time.sleep(60)
                
    def evaluate_strategy_performance(self):
        """Monitor strategy performance and trigger fallback if needed"""
        try:
            # Check for performance data
            message = self.performance_sub.recv_string(zmq.NOBLOCK)
            data = json.loads(message)
            
            # XGBoost confidence check
            model_confidence = data.get('model_confidence', 0)
            if model_confidence < 0.55 and self.current_strategy == "xgboost":
                self.switch_to_fallback("low_confidence")
                
            # Recent performance check (last 10 trades)
            recent_win_rate = data.get('recent_win_rate', 0)
            if recent_win_rate < 0.35 and self.current_strategy == "xgboost":
                self.switch_to_fallback("poor_performance")
                
            # Recovery check - switch back to XGBoost if conditions improve
            if (self.current_strategy == "fallback" and 
                model_confidence > 0.7 and 
                recent_win_rate > 0.5 and
                time.time() - self.performance_metrics['last_switch_time'] > 3600):  # 1 hour cooldown
                self.switch_to_xgboost()
                
        except zmq.Again:
            pass  # No new performance data
        except Exception as e:
            logging.error(f"Performance evaluation error: {e}")
            
    def switch_to_fallback(self, reason):
        """Switch from XGBoost to fallback strategy"""
        if self.current_strategy != "fallback":
            logging.warning(f"Switching to fallback strategy: {reason}")
            self.current_strategy = "fallback"
            self.fallback_triggered = True
            self.performance_metrics['last_switch_time'] = time.time()
            self.performance_metrics['switch_count'] += 1
            
            # Notify all services
            message = {
                'timestamp': datetime.now().isoformat(),
                'event': 'strategy_switch',
                'from': 'xgboost',
                'to': 'fallback',
                'reason': reason
            }
            self.strategy_pub.send_string(json.dumps(message))
            
    def switch_to_xgboost(self):
        """Revert back to XGBoost strategy"""
        if self.current_strategy != "xgboost":
            logging.info("Reverting to XGBoost strategy")
            self.current_strategy = "xgboost"
            self.performance_metrics['last_switch_time'] = time.time()
            
            message = {
                'timestamp': datetime.now().isoformat(),
                'event': 'strategy_switch', 
                'from': 'fallback',
                'to': 'xgboost',
                'reason': 'recovery'
            }
            self.strategy_pub.send_string(json.dumps(message))
            
    def get_status(self):
        """Return current orchestrator status"""
        return {
            'current_strategy': self.current_strategy,
            'fallback_triggered': self.fallback_triggered,
            'performance_metrics': self.performance_metrics
        }

