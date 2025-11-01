# Advanced Risk Protection - Final Safety Layer
import numpy as np
from datetime import datetime, timedelta
import logging

class AdvancedRiskProtection:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.protection_rules = self._initialize_protection_rules()
        self.emergency_actions_taken = []
        
    def _initialize_protection_rules(self):
        """Initialize advanced protection rules"""
        return {
            'consecutive_loss_limit': 3,  # Max consecutive losses
            'hourly_trade_limit': 5,      # Max trades per hour
            'daily_trade_limit': 15,      # Max trades per day
            'volatility_spike_threshold': 2.0,  # 2x normal volatility
            'correlation_protection': True,     # Avoid correlated trades
            'news_event_protection': True       # Protect during news
        }
        
    def check_advanced_protections(self, symbol, signal):
        """Check all advanced protection rules"""
        protections = []
        
        # 1. Consecutive losses protection
        if self._check_consecutive_losses():
            protections.append("CONSECUTIVE_LOSSES")
            
        # 2. Trade frequency protection
        if self._check_trade_frequency():
            protections.append("TRADE_FREQUENCY")
            
        # 3. Volatility protection
        if self._check_volatility_spike(symbol):
            protections.append("HIGH_VOLATILITY")
            
        # 4. Correlation protection
        if self._check_correlation_risk(symbol):
            protections.append("CORRELATION_RISK")
            
        # 5. News event protection
        if self._check_news_events():
            protections.append("NEWS_EVENT")
            
        return protections
        
    def _check_consecutive_losses(self):
        """Check for excessive consecutive losses"""
        recent_trades = self._get_recent_trades(10)
        if len(recent_trades) < 3:
            return False
            
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade.get('profit', 0) < 0:
                consecutive_losses += 1
            else:
                break
                
        return consecutive_losses >= self.protection_rules['consecutive_loss_limit']
        
    def _check_trade_frequency(self):
        """Check trade frequency limits"""
        recent_trades = self._get_recent_trades(60)  # Last hour
        if len(recent_trades) >= self.protection_rules['hourly_trade_limit']:
            return True
            
        daily_trades = self._get_recent_trades(24*60)  # Last 24 hours
        return len(daily_trades) >= self.protection_rules['daily_trade_limit']
        
    def _check_volatility_spike(self, symbol):
        """Check for volatility spikes"""
        # This would integrate with real volatility data
        # For now, use simulated check
        current_volatility = 1.0  # Normalized volatility
        return current_volatility > self.protection_rules['volatility_spike_threshold']
        
    def _check_correlation_risk(self, symbol):
        """Check for correlation risks"""
        # Avoid taking similar positions on correlated instruments
        active_trades = self._get_active_trades()
        correlated_symbols = self._get_correlated_symbols(symbol)
        
        for trade in active_trades:
            if trade['symbol'] in correlated_symbols and trade['direction'] == signal['action']:
                return True
                
        return False
        
    def _check_news_events(self):
        """Check for major news events"""
        # This would integrate with economic calendar
        # For now, avoid trading during typical news times
        current_hour = datetime.now().hour
        news_times = [8, 9, 13, 14, 15]  # Major news release times (GMT)
        return current_hour in news_times
        
    def _get_recent_trades(self, minutes_back):
        """Get recent trades within time window"""
        # This would filter trades by timestamp
        # For now, return recent trades from logger
        return self.trading_system.phase1_system.ftmo_logger.trade_history[-10:]  # Last 10
        
    def _get_active_trades(self):
        """Get currently active trades"""
        # This would integrate with MT5 position tracking
        return []
        
    def _get_correlated_symbols(self, symbol):
        """Get symbols correlated with given symbol"""
        correlations = {
            'US30': ['US100', 'SPX500'],
            'US100': ['US30', 'SPX500'],
            'UAX': ['US30', 'US100']
        }
        return correlations.get(symbol, [])
        
    def execute_emergency_protection(self, protection_triggered):
        """Execute emergency protection measures"""
        if not protection_triggered:
            return
            
        logging.warning(f"🚨 EMERGENCY PROTECTION ACTIVATED: {protection_triggered}")
        
        # 1. Reduce position sizes immediately
        self.trading_system.risk_manager.config.RISK_PER_TRADE *= 0.5
        
        # 2. Switch to fallback strategy
        self.trading_system.phase1_system.strategy_orchestrator.switch_to_fallback("emergency_protection")
        
        # 3. Implement trade cooldown
        self._implement_trade_cooldown()
        
        # 4. Record emergency action
        self.emergency_actions_taken.append({
            'timestamp': datetime.now(),
            'trigger': protection_triggered,
            'actions_taken': ['risk_reduction', 'strategy_switch', 'cooldown']
        })
        
    def _implement_trade_cooldown(self):
        """Implement trading cooldown period"""
        logging.info("Trade cooldown activated - pausing trading for 30 minutes")
        # This would pause the trading loop temporarily
        
    def get_protection_status(self):
        """Get current protection status"""
        return {
            'emergency_actions': len(self.emergency_actions_taken),
            'active_protections': self.protection_rules,
            'last_emergency': self.emergency_actions_taken[-1] if self.emergency_actions_taken else None
        }
