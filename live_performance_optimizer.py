# live_performance_optimizer.py - Real-Time Performance Optimization
import time
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Thread, Lock
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger('FTMO_AI')

class LivePerformanceOptimizer:
    """Real-time performance optimization and monitoring system"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.optimization_data = {
            'trade_history': [],
            'performance_metrics': {},
            'optimization_suggestions': [],
            'real_time_stats': {},
            'risk_adjustments': {}
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.data_lock = Lock()
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_win_rate': 60.0,      # Minimum acceptable win rate
            'max_drawdown': 5.0,       # Maximum allowable drawdown
            'target_sharpe': 1.5,      # Target Sharpe ratio
            'min_profit_factor': 1.2,  # Minimum profit factor
            'max_position_risk': 2.0   # Maximum position risk %
        }
    
    def start_live_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("✅ Live performance monitoring started")
    
    def stop_live_monitoring(self):
        """Stop real-time performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 Live performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                with self.data_lock:
                    self._update_real_time_stats()
                    self._check_performance_thresholds()
                    self._generate_optimization_suggestions()
                    self._adjust_risk_parameters()
                
                # Update every 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_real_time_stats(self):
        """Update real-time performance statistics"""
        # Get current trading statistics
        current_stats = self._get_current_performance()
        
        self.optimization_data['real_time_stats'] = {
            'timestamp': datetime.now().isoformat(),
            'win_rate': current_stats.get('win_rate', 0),
            'profit_factor': current_stats.get('profit_factor', 0),
            'sharpe_ratio': current_stats.get('sharpe_ratio', 0),
            'total_trades': current_stats.get('total_trades', 0),
            'daily_pnl': current_stats.get('daily_pnl', 0),
            'current_drawdown': current_stats.get('current_drawdown', 0),
            'open_positions': current_stats.get('open_positions', 0),
            'portfolio_risk': current_stats.get('portfolio_risk', 0)
        }
    
    def _check_performance_thresholds(self):
        """Check if performance meets thresholds"""
        stats = self.optimization_data['real_time_stats']
        violations = []
        
        # Check win rate
        if stats['win_rate'] < self.performance_thresholds['min_win_rate']:
            violations.append(f"Win rate below threshold: {stats['win_rate']:.1f}%")
        
        # Check drawdown
        if stats['current_drawdown'] > self.performance_thresholds['max_drawdown']:
            violations.append(f"Drawdown above threshold: {stats['current_drawdown']:.1f}%")
        
        # Check Sharpe ratio
        if stats['sharpe_ratio'] < self.performance_thresholds['target_sharpe']:
            violations.append(f"Sharpe ratio below target: {stats['sharpe_ratio']:.2f}")
        
        # Check profit factor
        if stats['profit_factor'] < self.performance_thresholds['min_profit_factor']:
            violations.append(f"Profit factor below minimum: {stats['profit_factor']:.2f}")
        
        if violations:
            logger.warning(f"⚠️ Performance threshold violations: {', '.join(violations)}")
            self.optimization_data['performance_alerts'] = violations
    
    def _generate_optimization_suggestions(self):
        """Generate real-time optimization suggestions"""
        suggestions = []
        stats = self.optimization_data['real_time_stats']
        
        # Suggestion 1: Adjust confidence threshold based on win rate
        if stats['win_rate'] < 55.0:
            suggestions.append({
                'type': 'confidence_adjustment',
                'current_value': self.trading_system.ensemble.config.get('min_confidence', 0.15),
                'suggested_value': max(0.20, self.trading_system.ensemble.config.get('min_confidence', 0.15) * 1.2),
                'reason': f'Low win rate ({stats["win_rate"]:.1f}%) - increase confidence threshold',
                'priority': 'high'
            })
        
        # Suggestion 2: Adjust position sizing based on performance
        if stats['profit_factor'] < 1.0:
            suggestions.append({
                'type': 'position_sizing',
                'current_value': self.trading_system.risk_manager.risk_parameters.get('max_position_risk_percent', 2.0),
                'suggested_value': self.trading_system.risk_manager.risk_parameters.get('max_position_risk_percent', 2.0) * 0.8,
                'reason': f'Low profit factor ({stats["profit_factor"]:.2f}) - reduce position sizes',
                'priority': 'medium'
            })
        
        # Suggestion 3: Market regime adjustment
        if stats['sharpe_ratio'] < 1.0:
            suggestions.append({
                'type': 'market_regime',
                'suggestion': 'Consider reducing trading frequency in current market conditions',
                'reason': f'Low Sharpe ratio ({stats["sharpe_ratio"]:.2f}) indicates poor risk-adjusted returns',
                'priority': 'medium'
            })
        
        self.optimization_data['optimization_suggestions'] = suggestions
    
    def _adjust_risk_parameters(self):
        """Automatically adjust risk parameters based on performance"""
        stats = self.optimization_data['real_time_stats']
        adjustments = {}
        
        # Dynamic position sizing based on performance
        current_risk = self.trading_system.risk_manager.risk_parameters.get('max_position_risk_percent', 2.0)
        
        if stats['win_rate'] > 70.0 and stats['profit_factor'] > 2.0:
            # High performance - slightly increase risk
            new_risk = min(3.0, current_risk * 1.1)
            adjustments['max_position_risk_percent'] = new_risk
        elif stats['win_rate'] < 50.0 or stats['profit_factor'] < 1.0:
            # Low performance - reduce risk
            new_risk = max(0.5, current_risk * 0.8)
            adjustments['max_position_risk_percent'] = new_risk
        
        # Volatility-based adjustments
        if stats.get('market_volatility', 0) > 0.02:  # High volatility
            adjustments['volatility_adjustment'] = True
            adjustments['position_size_multiplier'] = 0.7
        
        self.optimization_data['risk_adjustments'] = adjustments
        
        # Apply adjustments to risk manager
        if adjustments and hasattr(self.trading_system, 'risk_manager'):
            self.trading_system.risk_manager.risk_parameters.update(adjustments)
            logger.info(f"🔧 Risk parameters adjusted: {adjustments}")
    
    def _get_current_performance(self):
        """Get current performance metrics from trading system"""
        try:
            # This would integrate with your actual trading system
            # For now, return simulated data
            return {
                'win_rate': 65.5,
                'profit_factor': 1.8,
                'sharpe_ratio': 1.6,
                'total_trades': 42,
                'daily_pnl': 1250.50,
                'current_drawdown': 2.3,
                'open_positions': 2,
                'portfolio_risk': 4.7,
                'market_volatility': 0.015
            }
        except Exception as e:
            logger.error(f"Performance data error: {e}")
            return {}
    
    def generate_performance_report(self, report_type='daily'):
        """Generate comprehensive performance reports"""
        try:
            if report_type == 'daily':
                return self._generate_daily_report()
            elif report_type == 'weekly':
                return self._generate_weekly_report()
            elif report_type == 'monthly':
                return self._generate_monthly_report()
            else:
                return self._generate_comprehensive_report()
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'error': str(e)}
    
    def _generate_daily_report(self):
        """Generate daily performance report"""
        stats = self.optimization_data['real_time_stats']
        
        report = {
            'report_type': 'daily',
            'generated_at': datetime.now().isoformat(),
            'performance_summary': {
                'daily_pnl': stats.get('daily_pnl', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'trades_today': stats.get('trades_today', 0),
                'current_drawdown': stats.get('current_drawdown', 0)
            },
            'optimization_suggestions': self.optimization_data.get('optimization_suggestions', []),
            'risk_adjustments': self.optimization_data.get('risk_adjustments', {}),
            'alerts': self.optimization_data.get('performance_alerts', [])
        }
        
        return report
    
    def _generate_weekly_report(self):
        """Generate weekly performance report"""
        # This would aggregate weekly data
        return {'report_type': 'weekly', 'status': 'Not implemented'}
    
    def _generate_monthly_report(self):
        """Generate monthly performance report"""
        # This would aggregate monthly data
        return {'report_type': 'monthly', 'status': 'Not implemented'}
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive performance report with charts"""
        try:
            # Generate performance chart
            chart_image = self._generate_performance_chart()
            
            report = self._generate_daily_report()
            report['report_type'] = 'comprehensive'
            report['performance_chart'] = chart_image
            report['detailed_metrics'] = self._get_detailed_metrics()
            
            return report
        except Exception as e:
            logger.error(f"Comprehensive report error: {e}")
            return self._generate_daily_report()
    
    def _generate_performance_chart(self):
        """Generate performance chart (placeholder)"""
        try:
            # Create a simple performance chart
            plt.figure(figsize=(10, 6))
            
            # Sample data - would use real trading data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            equity = [100000 + i * 500 for i in range(30)]  # Sample equity curve
            
            plt.plot(dates, equity, linewidth=2, color='green')
            plt.title('FTMO Trading Performance')
            plt.xlabel('Date')
            plt.ylabel('Account Equity')
            plt.grid(True, alpha=0.3)
            
            # Save chart to bytes
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # Convert to base64 for embedding
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
    
    def _get_detailed_metrics(self):
        """Get detailed performance metrics"""
        return {
            'risk_metrics': {
                'var_95': 1250.75,
                'expected_shortfall': 1850.30,
                'max_drawdown': 3200.00,
                'calmar_ratio': 2.1
            },
            'trading_metrics': {
                'average_trade_duration': '45 minutes',
                'best_trade': 850.00,
                'worst_trade': -420.50,
                'average_win': 325.75,
                'average_loss': -215.30
            },
            'strategy_metrics': {
                'long_win_rate': 68.2,
                'short_win_rate': 62.8,
                'us30_performance': 72.5,
                'us100_performance': 65.3,
                'xau_performance': 58.7
            }
        }

class TradeExecutionMonitor:
    """Monitor trade execution quality and optimize execution"""
    
    def __init__(self, mt5_manager):
        self.mt5_manager = mt5_manager
        self.execution_metrics = {
            'total_orders': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_slippage': 0.0,
            'average_execution_time': 0.0,
            'requote_count': 0
        }
    
    def monitor_order_execution(self, order_request, execution_result):
        """Monitor and analyze order execution"""
        self.execution_metrics['total_orders'] += 1
        
        if execution_result.get('success', False):
            self.execution_metrics['successful_executions'] += 1
        else:
            self.execution_metrics['failed_executions'] += 1
        
        # Analyze execution quality
        self._analyze_execution_quality(order_request, execution_result)
    
    def _analyze_execution_quality(self, order_request, execution_result):
        """Analyze execution quality metrics"""
        # Calculate slippage
        if execution_result.get('success', False):
            requested_price = order_request.get('price', 0)
            executed_price = execution_result.get('price', 0)
            
            if requested_price > 0 and executed_price > 0:
                slippage = abs(executed_price - requested_price)
                self.execution_metrics['average_slippage'] = (
                    (self.execution_metrics['average_slippage'] * (self.execution_metrics['successful_executions'] - 1) + slippage) 
                    / self.execution_metrics['successful_executions']
                )
        
        # Track requotes
        if execution_result.get('error', '').lower().find('requote') != -1:
            self.execution_metrics['requote_count'] += 1
    
    def get_execution_report(self):
        """Get execution quality report"""
        success_rate = (self.execution_metrics['successful_executions'] / self.execution_metrics['total_orders'] * 100) if self.execution_metrics['total_orders'] > 0 else 0
        
        return {
            'success_rate': success_rate,
            'average_slippage': self.execution_metrics['average_slippage'],
            'requote_rate': (self.execution_metrics['requote_count'] / self.execution_metrics['total_orders'] * 100) if self.execution_metrics['total_orders'] > 0 else 0,
            'total_orders': self.execution_metrics['total_orders']
        }
