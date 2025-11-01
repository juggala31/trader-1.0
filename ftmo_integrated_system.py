# FTMO Trading System - Phase 1 Integration
# Main integration file to connect all Phase 1 components

import zmq
import json
import logging
import time
from datetime import datetime
from strategy_orchestrator import StrategyOrchestrator
from probability_calibrator import ProbabilityCalibrator
from model_loader import ModelLoader
from ftmo_challenge_logger import FTMOChallengeLogger
from service_health_monitor import ServiceHealthMonitor
from ftmo_config import FTMOConfig

class FTMOIntegratedSystem:
    def __init__(self, account_id="1600038177", challenge_type="200k"):
        self.account_id = account_id
        self.challenge_type = challenge_type
        
        # Initialize configuration
        self.config = FTMOConfig()
        
        # Initialize all Phase 1 components
        self.strategy_orchestrator = StrategyOrchestrator(self.config)
        self.model_loader = ModelLoader()
        self.model = self.model_loader.load_model()
        self.probability_calibrator = ProbabilityCalibrator(self.model_loader)
        self.ftmo_logger = FTMOChallengeLogger(account_id, challenge_type)
        self.health_monitor = ServiceHealthMonitor(self.config.SERVICES)
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Trading state
        self.is_running = False
        self.current_trades = []
        
        logging.info("FTMO Integrated System initialized")
        
    def start_trading(self):
        """Start the integrated trading system"""
        self.is_running = True
        logging.info("FTMO Trading System started")
        
        # Main trading loop
        while self.is_running:
            try:
                # Check system health first
                health_report = self.health_monitor.get_health_report()
                if health_report['overall_status'] == 'critical':
                    logging.error("System health critical - pausing trading")
                    time.sleep(60)
                    continue
                    
                # Execute trading cycle
                self._trading_cycle()
                
                # Brief pause between cycles
                time.sleep(10)
                
            except Exception as e:
                logging.error(f"Trading cycle error: {e}")
                time.sleep(30)
                
    def _trading_cycle(self):
        """Single trading cycle"""
        # Get current strategy
        current_strategy = self.strategy_orchestrator.current_strategy
        
        # Generate trading signal based on current strategy
        if current_strategy == "xgboost":
            signal = self._get_xgboost_signal()
        else:
            signal = self._get_fallback_signal()
            
        # Execute trade if signal is valid
        if signal and self._validate_signal(signal):
            trade_result = self._execute_trade(signal)
            
            # Log trade with FTMO system
            if trade_result:
                self.ftmo_logger.log_trade(trade_result)
                
    def _get_xgboost_signal(self):
        """Get signal from XGBoost strategy with probability calibration"""
        try:
            # Get features (you'll need to implement this based on your data)
            features = self._get_current_features()
            
            # Get prediction using the model loader
            if self.model_loader.is_model_loaded():
                raw_prediction = self.model.predict_proba([features])[0]
            else:
                # Use fallback prediction
                raw_prediction = [0.5, 0.5]  # Neutral fallback
                
            # Calibrate probabilities
            calibrated_probs, confidence = self.probability_calibrator.calibrate_probabilities(
                features, raw_prediction
            )
            
            # Only trade if confidence is high enough
            if confidence > self.config.XGBOOST_CONFIDENCE_THRESHOLD:
                direction = 1 if calibrated_probs[1] > calibrated_probs[0] else -1
                return {
                    'strategy': 'xgboost',
                    'direction': direction,
                    'confidence': confidence,
                    'symbol': self.config.DEFAULT_SYMBOL
                }
                
        except Exception as e:
            logging.error(f"XGBoost signal error: {e}")
            
        return None
        
    def _get_fallback_signal(self):
        """Get signal from fallback strategy"""
        try:
            # Implement your fallback strategy logic here
            # This should call your existing EMA/RSI strategy
            fallback_signal = self._call_fallback_strategy()
            
            return {
                'strategy': 'fallback',
                'direction': fallback_signal,
                'confidence': 0.8,  # Fallback has fixed confidence
                'symbol': self.config.DEFAULT_SYMBOL
            }
            
        except Exception as e:
            logging.error(f"Fallback signal error: {e}")
            
        return None
        
    def _validate_signal(self, signal):
        """Validate trading signal against FTMO rules"""
        # Check if we have too many concurrent trades
        if len(self.current_trades) >= self.config.MAX_CONCURRENT_TRADES:
            return False
            
        # Check FTMO daily loss limit (don't open new trades if near limit)
        daily_profit = self.ftmo_logger.metrics['daily_profit']
        daily_limit = self.ftmo_logger.metrics['daily_loss_limit']
        
        if daily_profit < -daily_limit * 0.8:  # 80% of limit
            logging.warning("Near daily loss limit - skipping trade")
            return False
            
        return True
        
    def _execute_trade(self, signal):
        """Execute a trade (integrate with your existing trade execution)"""
        try:
            # This should call your existing trade execution logic
            # For now, return simulated result
            profit = 50 if signal['direction'] == 1 else -30  # Simulated
            
            trade_data = {
                'symbol': signal['symbol'],
                'profit': profit,
                'type': 'buy' if signal['direction'] == 1 else 'sell',
                'strategy': signal['strategy'],
                'confidence': signal.get('confidence', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            
            self.current_trades.append(trade_data)
            logging.info(f"Trade executed: {trade_data}")
            
            return trade_data
            
        except Exception as e:
            logging.error(f"Trade execution error: {e}")
            return None
            
    def stop_trading(self):
        """Stop the trading system"""
        self.is_running = False
        logging.info("FTMO Trading System stopped")
        
    def get_system_status(self):
        """Get complete system status"""
        return {
            'trading_active': self.is_running,
            'current_strategy': self.strategy_orchestrator.current_strategy,
            'ftmo_metrics': self.ftmo_logger.metrics,
            'health_status': self.health_monitor.get_health_report(),
            'active_trades': len(self.current_trades)
        }

# Integration helper functions
def integrate_with_existing_system():
    """Helper function to integrate with your existing system"""
    
    # Replace your main trading loop with this integration
    system = FTMOIntegratedSystem()
    
    try:
        system.start_trading()
    except KeyboardInterrupt:
        system.stop_trading()
    except Exception as e:
        logging.error(f"System error: {e}")
        system.stop_trading()

if __name__ == "__main__":
    # Test the integrated system
    system = FTMOIntegratedSystem()
    print("Integrated System Status:", system.get_system_status())

