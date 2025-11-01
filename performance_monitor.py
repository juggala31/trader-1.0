# Real-Time Performance Monitor - Phase 3
import time
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from threading import Thread

class RealTimePerformanceMonitor:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.performance_data = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance metrics storage
        self.metrics_history = {
            'equity': [],
            'drawdown': [],
            'daily_profit': [],
            'win_rate': [],
            'risk_level': []
        }
        
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        self.monitoring_active = True
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Real-time performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("Performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get current system status
                status = self.trading_system.get_phase2_status()
                
                # Extract performance metrics
                metrics = status['phase1_status']['ftmo_metrics']
                risk_data = status['risk_manager']
                
                # Record metrics
                timestamp = datetime.now()
                self._record_metrics(timestamp, metrics, risk_data)
                
                # Check performance thresholds
                self._check_performance_alerts(metrics, risk_data)
                
                # Generate periodic reports
                if len(self.metrics_history['equity']) % 60 == 0:  # Every hour
                    self._generate_hourly_report()
                    
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(120)  # Wait longer on error
                
    def _record_metrics(self, timestamp, metrics, risk_data):
        """Record performance metrics"""
        record = {
            'timestamp': timestamp,
            'equity': metrics.get('current_balance', 0),
            'drawdown': risk_data.get('current_drawdown', 0),
            'daily_profit': metrics.get('daily_profit', 0),
            'total_profit': metrics.get('total_profit', 0),
            'total_trades': metrics.get('total_trades', 0),
            'winning_trades': metrics.get('winning_trades', 0),
            'risk_level': risk_data.get('risk_level', 'UNKNOWN')
        }
        
        self.performance_data.append(record)
        
        # Update history for charts
        self.metrics_history['equity'].append(record['equity'])
        self.metrics_history['drawdown'].append(record['drawdown'])
        self.metrics_history['daily_profit'].append(record['daily_profit'])
        
        # Calculate win rate
        if record['total_trades'] > 0:
            win_rate = record['winning_trades'] / record['total_trades']
        else:
            win_rate = 0
        self.metrics_history['win_rate'].append(win_rate)
        
        self.metrics_history['risk_level'].append(record['risk_level'])
        
    def _check_performance_alerts(self, metrics, risk_data):
        """Check for performance-based alerts"""
        # Profit target progress
        total_profit = metrics.get('total_profit', 0)
        profit_target = metrics.get('profit_target', 10000)
        
        if total_profit >= profit_target:
            self._trigger_performance_alert("SUCCESS", f"Profit target achieved: ${total_profit}")
            
        # Win rate monitoring
        total_trades = metrics.get('total_trades', 0)
        winning_trades = metrics.get('winning_trades', 0)
        
        if total_trades > 10:
            win_rate = winning_trades / total_trades
            if win_rate < 0.3:
                self._trigger_performance_alert("WARNING", f"Low win rate: {win_rate:.1%}")
                
        # Drawdown monitoring
        drawdown = risk_data.get('current_drawdown', 0)
        if drawdown > 0.05:  # 5% drawdown
            self._trigger_performance_alert("CRITICAL", f"High drawdown: {drawdown:.1%}")
            
    def _trigger_performance_alert(self, level, message):
        """Trigger performance alert"""
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        }
        
        logging.warning(f"PERFORMANCE ALERT {level}: {message}")
        
    def _generate_hourly_report(self):
        """Generate hourly performance report"""
        if len(self.performance_data) < 2:
            return
            
        # Calculate hourly metrics
        recent_data = self.performance_data[-60:]  # Last hour
        
        hourly_profit = recent_data[-1]['total_profit'] - recent_data[0]['total_profit']
        trades_per_hour = recent_data[-1]['total_trades'] - recent_data[0]['total_trades']
        
        report = {
            'timestamp': datetime.now(),
            'hourly_profit': hourly_profit,
            'trades_per_hour': trades_per_hour,
            'average_win_rate': self._calculate_average_win_rate(recent_data),
            'max_drawdown': max([d['drawdown'] for d in recent_data]),
            'risk_level_changes': len(set([d['risk_level'] for d in recent_data]))
        }
        
        logging.info(f"Hourly Report: Profit: ${hourly_profit:.2f}, Trades: {trades_per_hour}")
        
    def _calculate_average_win_rate(self, data):
        """Calculate average win rate from data"""
        if len(data) < 2:
            return 0
            
        start_trades = data[0]['total_trades']
        end_trades = data[-1]['total_trades']
        start_wins = data[0]['winning_trades']
        end_wins = data[-1]['winning_trades']
        
        if end_trades - start_trades > 0:
            return (end_wins - start_wins) / (end_trades - start_trades)
        return 0
        
    def generate_performance_dashboard(self):
        """Generate performance dashboard"""
        if len(self.performance_data) < 10:
            return "Not enough data for dashboard"
            
        df = pd.DataFrame(self.performance_data)
        
        dashboard = {
            'current_equity': df['equity'].iloc[-1],
            'peak_equity': df['equity'].max(),
            'current_drawdown': df['drawdown'].iloc[-1],
            'max_drawdown': df['drawdown'].max(),
            'total_profit': df['total_profit'].iloc[-1],
            'total_trades': df['total_trades'].iloc[-1],
            'win_rate': df['winning_trades'].iloc[-1] / df['total_trades'].iloc[-1] if df['total_trades'].iloc[-1] > 0 else 0,
            'current_risk_level': df['risk_level'].iloc[-1],
            'monitoring_duration': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600
        }
        
        return dashboard
        
    def export_performance_data(self, filename=None):
        """Export performance data to file"""
        if filename is None:
            filename = f"ftmo_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.performance_data, f, indent=2, default=str)
            
        logging.info(f"Performance data exported to {filename}")
        return filename
