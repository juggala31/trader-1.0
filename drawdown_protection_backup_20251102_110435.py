# Drawdown Protection System - Phase 2
import numpy as np
from datetime import datetime, timedelta
import logging

class DrawdownProtection:
    def __init__(self, ftmo_logger):
        self.ftmo_logger = ftmo_logger
        self.drawdown_history = []
        self.protection_active = False
        self.position_size_reduction = 1.0
        self.max_drawdown_threshold = 0.05  # 5% drawdown triggers protection
        
    def monitor_drawdown(self):
        """Monitor current drawdown and activate protection if needed"""
        current_equity = self.ftmo_logger.metrics.get('current_balance', 100000)
        peak_equity = self.ftmo_logger.metrics.get('peak_balance', current_equity)
        
        if peak_equity > 0:
            current_drawdown = (peak_equity - current_equity) / peak_equity
            
            # Record drawdown
            self.drawdown_history.append({
                'timestamp': datetime.now(),
                'drawdown': current_drawdown,
                'equity': current_equity,
                'peak': peak_equity
            })
            
            # Keep only last 100 records
            self.drawdown_history = self.drawdown_history[-100:]
            
            # Check if protection should be activated
            self._update_protection_status(current_drawdown)
            
            return current_drawdown
        return 0.0
        
    def _update_protection_status(self, current_drawdown):
        """Update protection status based on drawdown level"""
        if current_drawdown >= self.max_drawdown_threshold:
            if not self.protection_active:
                logging.warning(f"Drawdown protection activated: {current_drawdown:.2%}")
                self.protection_active = True
                
            # Calculate position size reduction based on drawdown severity
            severity = current_drawdown / self.max_drawdown_threshold
            self.position_size_reduction = max(0.1, 1.0 / severity)
            
        else:
            if self.protection_active:
                logging.info("Drawdown protection deactivated")
                self.protection_active = False
                self.position_size_reduction = 1.0
                
    def get_position_size_multiplier(self):
        """Get position size multiplier during drawdown protection"""
        return self.position_size_reduction if self.protection_active else 1.0
        
    def should_reduce_exposure(self):
        """Determine if exposure should be reduced"""
        if not self.protection_active:
            return False
            
        # Additional checks for severe drawdown
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.max_drawdown_threshold * 1.5:  # 7.5% drawdown
            return True
            
        return False
        
    def get_current_drawdown(self):
        """Get current drawdown percentage"""
        current_equity = self.ftmo_logger.metrics.get('current_balance', 100000)
        peak_equity = self.ftmo_logger.metrics.get('peak_balance', current_equity)
        
        if peak_equity > 0:
            return (peak_equity - current_equity) / peak_equity
        return 0.0
        
    def get_protection_status(self):
        """Get current protection status"""
        return {
            'protection_active': self.protection_active,
            'current_drawdown': self.get_current_drawdown(),
            'position_size_multiplier': self.get_position_size_multiplier(),
            'drawdown_threshold': self.max_drawdown_threshold,
            'history_count': len(self.drawdown_history)
        }
        
    def emergency_recovery_plan(self):
        """Execute emergency recovery plan during severe drawdown"""
        if self.get_current_drawdown() > self.max_drawdown_threshold * 2:  # 10% drawdown
            logging.critical("Executing emergency recovery plan")
            
            # Step 1: Reduce position sizes to minimum
            self.position_size_reduction = 0.1
            
            # Step 2: Switch to conservative strategy
            # This would trigger strategy orchestrator to use fallback
            
            # Step 3: Implement trade frequency limits
            return {
                'action': 'EMERGENCY_RECOVERY',
                'position_size_limit': 0.1,
                'strategy_mode': 'CONSERVATIVE',
                'max_trades_per_hour': 1
            }
            
        return {'action': 'NORMAL'}
