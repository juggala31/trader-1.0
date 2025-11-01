# FTMO Challenge Compliance Logger
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import zmq
import threading
import time

class FTMOChallengeLogger:
    def __init__(self, account_id, challenge_type="200k"):
        self.account_id = account_id
        self.challenge_type = challenge_type
        self.start_date = datetime.now()
        self.current_date = self.start_date.date()
        self.rules = self._load_ftmo_rules()
        
        # Trading metrics storage
        self.metrics = {
            'daily_profit': 0.0,
            'max_daily_loss': 0.0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': self.rules['starting_balance'],
            'total_trades': 0,
            'winning_trades': 0,
            'current_balance': self.rules['starting_balance'],
            'daily_loss_limit': self.rules['daily_loss_limit'],
            'max_drawdown_limit': self.rules['max_drawdown_limit'],
            'profit_target': self.rules['profit_target']
        }
        
        # Trade history
        self.trade_history = []
        
        # Violation tracking
        self.violations = []
        
        # ZMQ for real-time monitoring
        self.context = zmq.Context()
        self.metrics_pub = self.context.socket(zmq.PUB)
        self.metrics_pub.bind("tcp://*:5558")
        
        # Daily reset thread
        self.reset_thread = threading.Thread(target=self._daily_reset_monitor, daemon=True)
        self.reset_thread.start()
        
        logging.info(f"FTMO Challenge Logger initialized for {challenge_type} challenge")
        
    def _load_ftmo_rules(self):
        """Load FTMO challenge specific rules based on challenge type"""
        rules = {
            '200k': {
                'starting_balance': 200000,
                'daily_loss_limit': 5000,
                'max_drawdown_limit': 10000,
                'profit_target': 10000,
                'min_trading_days': 5,
                'max_trading_period': 30
            },
            '100k': {
                'starting_balance': 100000,
                'daily_loss_limit': 2500,
                'max_drawdown_limit': 5000,
                'profit_target': 5000,
                'min_trading_days': 5,
                'max_trading_period': 30
            }
        }
        return rules.get(self.challenge_type, rules['200k'])
    
    def _daily_reset_monitor(self):
        """Monitor for daily reset (5 PM ET)"""
        while True:
            now = datetime.now()
            if now.hour == 17 and now.minute == 0:  # 5 PM
                self._reset_daily_metrics()
            time.sleep(60)  # Check every minute
            
    def _reset_daily_metrics(self):
        """Reset daily metrics at 5 PM ET"""
        self.metrics['daily_profit'] = 0.0
        self.metrics['max_daily_loss'] = 0.0
        self.current_date = datetime.now().date()
        logging.info("Daily metrics reset for FTMO challenge")
        
    def log_trade(self, trade_data):
        """Log each trade and check FTMO compliance"""
        self.metrics['total_trades'] += 1
        profit = trade_data.get('profit', 0)
        
        if profit > 0:
            self.metrics['winning_trades'] += 1
            
        self.metrics['daily_profit'] += profit
        self.metrics['total_profit'] += profit
        self.metrics['current_balance'] += profit
        
        # Update peak balance
        if self.metrics['current_balance'] > self.metrics['peak_balance']:
            self.metrics['peak_balance'] = self.metrics['current_balance']
            
        # Check daily loss limit
        if self.metrics['daily_profit'] < -self.metrics['daily_loss_limit']:
            self._trigger_violation("daily_loss_limit", 
                                   f"Daily loss: {self.metrics['daily_profit']}")
            
        # Update and check max drawdown
        self._update_drawdown()
        
        # Store trade history
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trade_history.append(trade_data)
        
        # Publish metrics for GUI
        self._publish_metrics()
        
        logging.info(f"Trade logged: {profit} | Daily P/L: {self.metrics['daily_profit']}")
        
    def _update_drawdown(self):
        """Calculate and update maximum drawdown"""
        if self.metrics['peak_balance'] > 0:
            drawdown = (self.metrics['peak_balance'] - self.metrics['current_balance']) 
            drawdown_percent = (drawdown / self.metrics['peak_balance']) * 100
            
            if drawdown > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = drawdown
                
            if drawdown > self.metrics['max_drawdown_limit']:
                self._trigger_violation("max_drawdown", 
                                       f"Drawdown: {drawdown} ({drawdown_percent:.1f}%)")
                
    def _trigger_violation(self, rule_type, details):
        """Record FTMO rule violation"""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'rule_type': rule_type,
            'details': details,
            'current_metrics': self.metrics.copy()
        }
        self.violations.append(violation)
        logging.error(f"FTMO VIOLATION: {rule_type} - {details}")
        
    def _publish_metrics(self):
        """Publish current metrics via ZMQ"""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'violation_count': len(self.violations),
            'challenge_progress': self._calculate_progress()
        }
        self.metrics_pub.send_string(json.dumps(metrics_data))
        
    def _calculate_progress(self):
        """Calculate challenge progress percentage"""
        days_passed = (datetime.now().date() - self.start_date.date()).days
        max_days = self.rules['max_trading_period']
        profit_progress = min(self.metrics['total_profit'] / self.rules['profit_target'], 1.0)
        
        return {
            'days_passed': days_passed,
            'days_required': self.rules['min_trading_days'],
            'profit_progress': profit_progress,
            'overall_progress': (days_passed / max_days + profit_progress) / 2
        }
        
    def get_challenge_status(self):
        """Return complete challenge status"""
        return {
            'metrics': self.metrics,
            'violations': self.violations,
            'progress': self._calculate_progress(),
            'rules': self.rules
        }
