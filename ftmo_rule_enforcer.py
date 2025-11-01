# FTMO Rule Enforcer - Automated Rule Compliance
import time
from datetime import datetime, timedelta
import logging

class FTMO_Rule_Enforcer:
    def __init__(self, ftmo_logger):
        self.ftmo_logger = ftmo_logger
        self.rules_violated = []
        self.trading_halted = False
        self.halt_reason = ""
        self.rule_checks = {
            'daily_loss_limit': self._check_daily_loss_limit,
            'max_drawdown_limit': self._check_max_drawdown_limit,
            'min_trading_days': self._check_min_trading_days,
            'profit_target': self._check_profit_target,
            'consecutive_losses': self._check_consecutive_losses
        }
        
    def enforce_rules(self):
        """Enforce all FTMO rules"""
        violations = []
        
        for rule_name, check_function in self.rule_checks.items():
            is_violated, message = check_function()
            if is_violated:
                violations.append({'rule': rule_name, 'message': message})
                self._handle_violation(rule_name, message)
                
        return violations
        
    def _check_daily_loss_limit(self):
        """Check daily loss limit rule"""
        daily_profit = self.ftmo_logger.metrics.get('daily_profit', 0)
        daily_limit = self.ftmo_logger.metrics.get('daily_loss_limit', 5000)
        
        if daily_profit <= -daily_limit:
            return True, f"Daily loss limit violated: {daily_profit} <= {-daily_limit}"
        return False, ""
        
    def _check_max_drawdown_limit(self):
        """Check maximum drawdown rule"""
        max_drawdown = self.ftmo_logger.metrics.get('max_drawdown', 0)
        drawdown_limit = self.ftmo_logger.metrics.get('max_drawdown_limit', 10000)
        
        if max_drawdown >= drawdown_limit:
            return True, f"Max drawdown limit violated: {max_drawdown} >= {drawdown_limit}"
        return False, ""
        
    def _check_min_trading_days(self):
        """Check minimum trading days rule"""
        start_date = self.ftmo_logger.start_date
        days_passed = (datetime.now().date() - start_date.date()).days
        min_days = self.ftmo_logger.rules.get('min_trading_days', 5)
        
        if days_passed < min_days and self.ftmo_logger.metrics.get('total_profit', 0) >= self.ftmo_logger.rules.get('profit_target', 10000):
            return True, f"Profit target reached before minimum trading days: {days_passed} < {min_days}"
        return False, ""
        
    def _check_profit_target(self):
        """Check profit target rule (positive check)"""
        total_profit = self.ftmo_logger.metrics.get('total_profit', 0)
        profit_target = self.ftmo_logger.rules.get('profit_target', 10000)
        
        if total_profit >= profit_target:
            return True, f"Profit target achieved: {total_profit} >= {profit_target}"
        return False, ""
        
    def _check_consecutive_losses(self):
        """Check for excessive consecutive losses"""
        recent_trades = self.ftmo_logger.trade_history[-10:]  # Last 10 trades
        if len(recent_trades) < 5:
            return False, ""
            
        consecutive_losses = 0
        max_consecutive = 0
        
        for trade in reversed(recent_trades):
            if trade.get('profit', 0) < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
                
        if max_consecutive >= 5:  # 5 consecutive losses
            return True, f"Excessive consecutive losses: {max_consecutive} in a row"
        return False, ""
        
    def _handle_violation(self, rule_name, message):
        """Handle rule violation"""
        if rule_name in ['daily_loss_limit', 'max_drawdown_limit']:
            # Critical violations - halt trading
            self.trading_halted = True
            self.halt_reason = message
            logging.critical(f"TRADING HALTED: {message}")
            
        elif rule_name == 'profit_target':
            # Positive violation - target achieved
            logging.info(f"FTMO TARGET ACHIEVED: {message}")
            
        # Record violation
        violation = {
            'timestamp': datetime.now(),
            'rule': rule_name,
            'message': message,
            'metrics': self.ftmo_logger.metrics.copy()
        }
        self.rules_violated.append(violation)
        
    def is_trading_allowed(self):
        """Check if trading is currently allowed"""
        if self.trading_halted:
            return False, self.halt_reason
            
        # Check if market hours (basic check) - Extended for testing
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour >= 22:  # 6 AM to 10 PM for testing
            return False, "Outside trading hours"
            
        return True, "Trading allowed"
        
    def get_enforcement_status(self):
        """Get current enforcement status"""
        return {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'violations_count': len(self.rules_violated),
            'last_check': datetime.now().isoformat(),
            'rules_checked': list(self.rule_checks.keys())
        }
        
    def resume_trading(self):
        """Resume trading after halt (manual override)"""
        if self.trading_halted:
            logging.warning("Trading resumed by manual override")
            self.trading_halted = False
            self.halt_reason = ""
            return True
        return False

