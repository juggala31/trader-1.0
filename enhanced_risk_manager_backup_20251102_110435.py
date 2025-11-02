# Enhanced Risk Manager - Phase 2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from ftmo_config_complete import FTMOConfig

class EnhancedRiskManager:
    def __init__(self, ftmo_logger):
        self.config = FTMOConfig()
        self.ftmo_logger = ftmo_logger
        self.risk_level = "NORMAL"  # NORMAL, CAUTION, HIGH_RISK, LOCKDOWN
        self.volatility_adjustments = {}
        self.trade_history = []
        self.performance_metrics = {}
        
    def calculate_dynamic_position_size(self, symbol, signal_confidence, current_volatility):
        """Calculate dynamic position size based on multiple factors"""
        base_size = self._get_base_position_size()
        
        # Factor 1: Strategy Confidence
        confidence_factor = signal_confidence / 0.7  # Normalize to 70% confidence baseline
        
        # Factor 2: Volatility Adjustment
        volatility_factor = self._calculate_volatility_factor(current_volatility)
        
        # Factor 3: Account Equity Protection
        equity_factor = self._calculate_equity_protection_factor()
        
        # Factor 4: Drawdown Protection
        drawdown_factor = self._calculate_drawdown_factor()
        
        # Factor 5: Recent Performance
        performance_factor = self._calculate_performance_factor()
        
        # Combine all factors
        combined_factor = (confidence_factor * volatility_factor * 
                          equity_factor * drawdown_factor * performance_factor)
        
        # Apply risk level multiplier
        risk_multiplier = self._get_risk_level_multiplier()
        final_factor = combined_factor * risk_multiplier
        
        # Ensure reasonable bounds
        final_factor = max(0.1, min(2.0, final_factor))
        
        dynamic_size = base_size * final_factor
        
        logging.info(f"Position size: {dynamic_size:.4f} (Base: {base_size:.4f}, Factor: {final_factor:.2f})")
        return dynamic_size
        
    def _get_base_position_size(self):
        """Calculate base position size based on account equity and risk per trade"""
        account_equity = self.ftmo_logger.metrics.get('current_balance', 100000)
        risk_amount = account_equity * self.config.RISK_PER_TRADE
        return risk_amount / 1000  # Simplified calculation
        
    def _calculate_volatility_factor(self, volatility):
        """Adjust position size based on market volatility"""
        # Normalize volatility (assuming 1.0 is average)
        if volatility > 2.0:
            return 0.5  # High volatility - reduce size
        elif volatility > 1.5:
            return 0.7
        elif volatility < 0.5:
            return 1.3  # Low volatility - increase size
        else:
            return 1.0  # Normal volatility
            
    def _calculate_equity_protection_factor(self):
        """Protect account equity by reducing size during losses"""
        daily_profit = self.ftmo_logger.metrics.get('daily_profit', 0)
        daily_limit = self.ftmo_logger.metrics.get('daily_loss_limit', 5000)
        
        # Reduce position size as we approach daily loss limit
        if daily_profit < -daily_limit * 0.8:  # 80% of limit reached
            return 0.3
        elif daily_profit < -daily_limit * 0.5:  # 50% of limit reached
            return 0.6
        else:
            return 1.0
            
    def _calculate_drawdown_factor(self):
        """Adjust position size based on current drawdown"""
        max_drawdown = self.ftmo_logger.metrics.get('max_drawdown', 0)
        drawdown_limit = self.ftmo_logger.metrics.get('max_drawdown_limit', 10000)
        
        drawdown_ratio = max_drawdown / drawdown_limit if drawdown_limit > 0 else 0
        
        if drawdown_ratio > 0.8:
            return 0.2  # Severe drawdown - significantly reduce size
        elif drawdown_ratio > 0.5:
            return 0.5
        elif drawdown_ratio > 0.3:
            return 0.8
        else:
            return 1.0
            
    def _calculate_performance_factor(self):
        """Adjust based on recent trading performance"""
        recent_trades = self._get_recent_trades(20)  # Last 20 trades
        if len(recent_trades) < 5:
            return 1.0  # Not enough data
            
        winning_trades = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
        win_rate = winning_trades / len(recent_trades)
        
        if win_rate < 0.3:
            return 0.5  # Poor performance - reduce size
        elif win_rate < 0.4:
            return 0.7
        elif win_rate > 0.6:
            return 1.3  # Excellent performance - increase size
        else:
            return 1.0
            
    def _get_risk_level_multiplier(self):
        """Get multiplier based on current risk level"""
        multipliers = {
            "NORMAL": 1.0,
            "CAUTION": 0.7,
            "HIGH_RISK": 0.3,
            "LOCKDOWN": 0.0  # No trading
        }
        return multipliers.get(self.risk_level, 1.0)
        
    def update_risk_level(self):
        """Update overall risk level based on system conditions"""
        # Check daily loss limit
        daily_profit = self.ftmo_logger.metrics.get('daily_profit', 0)
        daily_limit = self.ftmo_logger.metrics.get('daily_loss_limit', 5000)
        
        # Check drawdown
        max_drawdown = self.ftmo_logger.metrics.get('max_drawdown', 0)
        drawdown_limit = self.ftmo_logger.metrics.get('max_drawdown_limit', 10000)
        
        if daily_profit <= -daily_limit or max_drawdown >= drawdown_limit:
            self.risk_level = "LOCKDOWN"
        elif daily_profit <= -daily_limit * 0.8 or max_drawdown >= drawdown_limit * 0.8:
            self.risk_level = "HIGH_RISK"
        elif daily_profit <= -daily_limit * 0.5 or max_drawdown >= drawdown_limit * 0.5:
            self.risk_level = "CAUTION"
        else:
            self.risk_level = "NORMAL"
            
        logging.info(f"Risk level updated: {self.risk_level}")
        
    def should_trade_be_executed(self, symbol, signal_type):
        """Determine if a trade should be executed based on risk rules"""
        if self.risk_level == "LOCKDOWN":
            return False, "Trading locked down due to risk limits"
            
        # Check concurrent trades limit
        active_trades = self._get_active_trades()
        if len(active_trades) >= self.config.MAX_CONCURRENT_TRADES:
            return False, "Maximum concurrent trades reached"
            
        # Check symbol-specific limits
        symbol_trades = [t for t in active_trades if t.get('symbol') == symbol]
        if len(symbol_trades) >= 2:  # Max 2 trades per symbol
            return False, "Maximum trades for this symbol reached"
            
        return True, "Trade approved"
        
    def _get_recent_trades(self, count=20):
        """Get recent trades from FTMO logger"""
        return self.ftmo_logger.trade_history[-count:] if self.ftmo_logger.trade_history else []
        
    def _get_active_trades(self):
        """Get currently active trades (simplified)"""
        # This would integrate with your MT5 position tracking
        return []  # Placeholder
        
    def get_risk_report(self):
        """Generate comprehensive risk report"""
        return {
            'risk_level': self.risk_level,
            'daily_profit': self.ftmo_logger.metrics.get('daily_profit', 0),
            'max_drawdown': self.ftmo_logger.metrics.get('max_drawdown', 0),
            'position_size_multiplier': self._get_risk_level_multiplier(),
            'trade_approval_rate': self._calculate_trade_approval_rate(),
            'timestamp': datetime.now().isoformat()
        }
        
    def _calculate_trade_approval_rate(self):
        """Calculate what percentage of trades would be approved"""
        # Simplified calculation based on risk level
        approval_rates = {
            "NORMAL": 1.0,
            "CAUTION": 0.7,
            "HIGH_RISK": 0.3,
            "LOCKDOWN": 0.0
        }
        return approval_rates.get(self.risk_level, 1.0)
