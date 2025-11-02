# FTMO Challenge-Specific Optimizations
import numpy as np
from datetime import datetime, time
import logging

class FTMOChallengeOptimizer:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.ftmo_rules = self._load_ftmo_rules()
        self.challenge_metrics = {}
        
    def _load_ftmo_rules(self):
        """Load specific FTMO challenge rules"""
        return {
            '200k': {
                'profit_target': 10000,
                'daily_loss_limit': 5000,
                'max_drawdown': 10000,
                'min_trading_days': 5,
                'max_trading_days': 30,
                'trading_instruments': ['Forex', 'Indices', 'Commodities']
            },
            '100k': {
                'profit_target': 5000,
                'daily_loss_limit': 2500,
                'max_drawdown': 5000,
                'min_trading_days': 5,
                'max_trading_days': 30,
                'trading_instruments': ['Forex', 'Indices', 'Commodities']
            }
        }
    
    def optimize_for_challenge(self, challenge_type="200k"):
        """Apply challenge-specific optimizations"""
        rules = self.ftmo_rules.get(challenge_type, self.ftmo_rules['200k'])
        
        # 1. Aggressive but controlled profit targeting
        self._set_aggressive_profit_strategy(rules)
        
        # 2. Time-based trading optimization
        self._optimize_trading_times()
        
        # 3. Volatility-based position sizing
        self._enhance_volatility_management()
        
        # 4. Challenge timeline optimization
        self._optimize_challenge_timeline(rules)
        
        logging.info(f"FTMO {challenge_type} challenge optimizations applied")
        
    def _set_aggressive_profit_strategy(self, rules):
        """Set more aggressive but controlled strategy for challenge"""
        # Increase trading frequency during high-probability periods
        # But maintain strict risk control
        
        # Adjust risk parameters for challenge
        original_risk = 0.02  # 2%
        challenge_risk = 0.025  # 2.5% - slightly more aggressive
        
        # But reduce size during uncertain periods
        self.trading_system.risk_manager.config.RISK_PER_TRADE = challenge_risk
        
        # Adjust confidence thresholds
        self.trading_system.phase1_system.config.XGBOOST_CONFIDENCE_THRESHOLD = 0.55  # Lower threshold for more trades
        self.trading_system.phase1_system.config.FALLBACK_TRIGGER_WIN_RATE = 0.30  # More tolerant before switching
        
    def _optimize_trading_times(self):
        """Optimize trading times for maximum efficiency"""
        # FTMO challenge requires efficient use of trading days
        self.optimal_trading_hours = {
            'london_open': time(8, 0),    # 8 AM GMT
            'us_open': time(13, 30),      # 1:30 PM GMT  
            'london_us_overlap': time(13, 0),  # 1-4 PM GMT (high volatility)
            'high_volatility_end': time(16, 0) # 4 PM GMT
        }
        
        # Focus trading during high-probability periods
        self.trading_system.phase1_system.config.PREFERRED_TRADING_HOURS = [
            (8, 12),   # London session
            (13, 16)   # US/London overlap
        ]
        
    def _enhance_volatility_management(self):
        """Enhanced volatility management for challenge"""
        # More dynamic ATR-based position sizing
        self.trading_system.risk_manager.config.VOLATILITY_ADJUSTMENT_FACTOR = 1.2
        
        # Adaptive SL/TP based on market conditions
        self.trading_system.phase1_system.config.ADAPTIVE_SL_TP = True
        
    def _optimize_challenge_timeline(self, rules):
        """Optimize strategy based on challenge timeline"""
        days_passed = self._get_challenge_days_passed()
        total_days = rules['max_trading_days']
        
        # Phase 1: First 5 days - Conservative
        if days_passed <= 5:
            self._set_conservative_mode()
            
        # Phase 2: Days 6-20 - Aggressive
        elif days_passed <= 20:
            self._set_aggressive_mode()
            
        # Phase 3: Final 10 days - Target-focused
        else:
            self._set_target_mode()
            
    def _set_conservative_mode(self):
        """Conservative mode for first 5 days"""
        logging.info("Challenge Phase 1: Conservative Mode")
        self.trading_system.risk_manager.config.RISK_PER_TRADE = 0.015  # 1.5%
        
    def _set_aggressive_mode(self):
        """Aggressive mode for middle phase"""
        logging.info("Challenge Phase 2: Aggressive Mode")
        self.trading_system.risk_manager.config.RISK_PER_TRADE = 0.025  # 2.5%
        
    def _set_target_mode(self):
        """Target-focused mode for final phase"""
        logging.info("Challenge Phase 3: Target Mode")
        # Adjust based on progress toward target
        progress = self._get_progress_toward_target()
        
        if progress >= 0.8:  # 80% toward target
            self.trading_system.risk_manager.config.RISK_PER_TRADE = 0.01  # 1% - protect gains
        else:
            self.trading_system.risk_manager.config.RISK_PER_TRADE = 0.03  # 3% - push for target
            
    def _get_challenge_days_passed(self):
        """Calculate days passed in challenge"""
        start_date = self.trading_system.phase1_system.ftmo_logger.start_date
        return (datetime.now().date() - start_date.date()).days
        
    def _get_progress_toward_target(self):
        """Calculate progress toward profit target"""
        metrics = self.trading_system.get_phase2_status()['phase1_status']['ftmo_metrics']
        current_profit = metrics.get('total_profit', 0)
        target_profit = metrics.get('profit_target', 10000)
        
        return current_profit / target_profit if target_profit > 0 else 0
        
    def get_challenge_strategy(self):
        """Get current challenge strategy"""
        days_passed = self._get_challenge_days_passed()
        progress = self._get_progress_toward_target()
        
        return {
            'days_passed': days_passed,
            'progress_percentage': progress * 100,
            'current_phase': self._get_current_phase(days_passed),
            'recommended_risk': self.trading_system.risk_manager.config.RISK_PER_TRADE,
            'trading_intensity': self._calculate_trading_intensity(days_passed, progress)
        }
        
    def _get_current_phase(self, days_passed):
        if days_passed <= 5:
            return "CONSERVATIVE"
        elif days_passed <= 20:
            return "AGGRESSIVE"
        else:
            return "TARGET_FOCUSED"
            
    def _calculate_trading_intensity(self, days_passed, progress):
        """Calculate optimal trading intensity"""
        if days_passed <= 5:
            return "LOW"  # Build confidence
        elif progress >= 0.8:
            return "LOW"  # Protect gains
        elif days_passed >= 25:
            return "HIGH" # Final push
        else:
            return "MEDIUM"
