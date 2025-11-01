# advanced_risk_management.py - Professional Position Sizing & Risk Management
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.stats import norm
import math

logger = logging.getLogger('FTMO_AI')

class ProfessionalRiskManager:
    """Professional risk management and position sizing system"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_parameters = {
            'max_daily_loss_percent': 5.0,      # FTMO daily loss limit
            'max_total_drawdown_percent': 10.0, # FTMO total drawdown limit
            'max_position_risk_percent': 2.0,   # Max risk per trade
            'max_portfolio_risk_percent': 15.0, # Max portfolio risk
            'target_risk_reward_ratio': 2.0,    # Minimum R:R ratio
            'volatility_adjustment': True,      # Adjust for market volatility
            'correlation_penalty': True,        # Penalize correlated positions
            'kelly_criterion_enabled': True,    # Use Kelly criterion for sizing
        }
        
        # Risk tracking
        self.risk_metrics = {
            'daily_pnl': 0.0,
            'total_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_balance': initial_balance,
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0
        }
        
        # Position tracking
        self.open_positions = {}
        self.position_correlations = {}
        self.trade_history = []
        
        # Market data
        self.volatility_data = {}
        self.correlation_matrix = {}
    
    def calculate_optimal_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                                      confidence: float, market_regime: int, portfolio_correlation: float) -> Dict:
        """Calculate optimal position size using multiple methodologies"""
        try:
            # Base position size using multiple methods
            kelly_size = self._kelly_position_size(confidence, entry_price, stop_loss)
            volatility_size = self._volatility_adjusted_size(symbol, entry_price, stop_loss, market_regime)
            risk_based_size = self._risk_based_size(entry_price, stop_loss)
            
            # Combine methodologies with weights
            combined_size = self._combine_position_sizes(
                kelly_size, volatility_size, risk_based_size, 
                confidence, market_regime
            )
            
            # Apply correlation penalty
            if self.risk_parameters['correlation_penalty']:
                combined_size *= self._calculate_correlation_penalty(portfolio_correlation)
            
            # Apply FTMO constraints
            constrained_size = self._apply_ftmo_constraints(combined_size, symbol)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_trade_risk_metrics(
                constrained_size, entry_price, stop_loss, symbol
            )
            
            return {
                'optimal_size': constrained_size,
                'risk_percent': risk_metrics['risk_percent'],
                'potential_loss': risk_metrics['potential_loss'],
                'potential_reward': risk_metrics['potential_reward'],
                'risk_reward_ratio': risk_metrics['risk_reward_ratio'],
                'methodology_breakdown': {
                    'kelly_size': kelly_size,
                    'volatility_size': volatility_size,
                    'risk_based_size': risk_based_size,
                    'combined_size': combined_size
                },
                'risk_metrics': risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Position sizing error for {symbol}: {e}")
            return self._get_fallback_position_size()
    
    def _kelly_position_size(self, confidence: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size using Kelly Criterion"""
        if not self.risk_parameters['kelly_criterion_enabled']:
            return 0.02  # Default 2% position size
        
        try:
            # Win probability (confidence adjusted)
            win_prob = max(0.1, min(0.9, confidence))
            
            # Loss probability
            loss_prob = 1.0 - win_prob
            
            # Calculate win/loss ratio (simplified)
            price_range = abs(entry_price - stop_loss)
            if price_range == 0:
                return 0.01  # Fallback
            
            # Assume 2:1 reward ratio as target
            win_amount = price_range * 2
            loss_amount = price_range
            
            # Kelly formula: f = (bp - q) / b
            # where b = win_amount/loss_amount, p = win_prob, q = loss_prob
            b = win_amount / loss_amount
            kelly_fraction = (b * win_prob - loss_prob) / b
            
            # Conservative Kelly (half-Kelly)
            conservative_kelly = kelly_fraction * 0.5
            
            # Ensure reasonable bounds
            return max(0.005, min(0.1, conservative_kelly))
            
        except Exception as e:
            logger.warning(f"Kelly calculation error: {e}")
            return 0.02  # Fallback
    
    def _volatility_adjusted_size(self, symbol: str, entry_price: float, stop_loss: float, market_regime: int) -> float:
        """Calculate position size adjusted for market volatility"""
        try:
            # Get current volatility for the symbol
            volatility = self._get_current_volatility(symbol)
            if volatility is None:
                volatility = 0.01  # Default 1% volatility
            
            # Base position size
            base_size = 0.02  # 2% base
            
            # Adjust for volatility regime
            regime_multipliers = {
                1: 1.2,  # Low volatility - increase size
                2: 1.0,  # Medium volatility - neutral
                3: 0.7   # High volatility - reduce size
            }
            regime_multiplier = regime_multipliers.get(market_regime, 1.0)
            
            # Adjust for specific volatility levels
            if volatility > 0.03:  # High volatility
                volatility_multiplier = 0.6
            elif volatility < 0.005:  # Low volatility
                volatility_multiplier = 1.3
            else:
                volatility_multiplier = 1.0
            
            adjusted_size = base_size * regime_multiplier * volatility_multiplier
            return max(0.005, min(0.1, adjusted_size))
            
        except Exception as e:
            logger.warning(f"Volatility adjustment error: {e}")
            return 0.02
    
    def _risk_based_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk parameters"""
        try:
            # Calculate price risk
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                return 0.01
            
            # Calculate maximum allowable loss
            max_loss_amount = self.current_balance * (self.risk_parameters['max_position_risk_percent'] / 100)
            
            # Position size based on risk
            risk_based_size = max_loss_amount / price_risk
            
            # Convert to percentage of account
            size_percent = (risk_based_size * entry_price) / self.current_balance
            
            return max(0.005, min(0.1, size_percent))
            
        except Exception as e:
            logger.warning(f"Risk-based sizing error: {e}")
            return 0.02
    
    def _combine_position_sizes(self, kelly_size: float, volatility_size: float, 
                               risk_based_size: float, confidence: float, market_regime: int) -> float:
        """Intelligently combine different position sizing methodologies"""
        # Weighting based on confidence and market regime
        if confidence > 0.7:  # High confidence
            weights = {'kelly': 0.5, 'volatility': 0.3, 'risk': 0.2}
        elif confidence > 0.4:  # Medium confidence
            weights = {'kelly': 0.3, 'volatility': 0.4, 'risk': 0.3}
        else:  # Low confidence
            weights = {'kelly': 0.2, 'volatility': 0.3, 'risk': 0.5}
        
        # Adjust weights for market regime
        if market_regime == 3:  # High volatility - favor risk-based
            weights = {'kelly': 0.2, 'volatility': 0.3, 'risk': 0.5}
        elif market_regime == 1:  # Low volatility - favor Kelly
            weights = {'kelly': 0.5, 'volatility': 0.3, 'risk': 0.2}
        
        # Calculate weighted average
        combined_size = (kelly_size * weights['kelly'] + 
                        volatility_size * weights['volatility'] + 
                        risk_based_size * weights['risk'])
        
        return combined_size
    
    def _calculate_correlation_penalty(self, portfolio_correlation: float) -> float:
        """Apply penalty for correlated positions"""
        if not self.risk_parameters['correlation_penalty']:
            return 1.0
        
        # Strong negative correlation -> increase size
        if portfolio_correlation < -0.7:
            return 1.2
        # Strong positive correlation -> reduce size
        elif portfolio_correlation > 0.7:
            return 0.6
        # Moderate correlation -> slight adjustment
        elif portfolio_correlation > 0.3:
            return 0.8
        elif portfolio_correlation < -0.3:
            return 1.1
        else:
            return 1.0
    
    def _apply_ftmo_constraints(self, position_size: float, symbol: str) -> float:
        """Apply FTMO-specific constraints to position size"""
        # Daily loss constraint
        daily_loss_limit = self.initial_balance * (self.risk_parameters['max_daily_loss_percent'] / 100)
        remaining_daily_risk = daily_loss_limit - self.risk_metrics['daily_pnl']
        
        if remaining_daily_risk <= 0:
            logger.warning("⚠️ Daily loss limit reached - position size reduced to zero")
            return 0.0
        
        # Total drawdown constraint
        total_drawdown_limit = self.initial_balance * (self.risk_parameters['max_total_drawdown_percent'] / 100)
        remaining_total_risk = total_drawdown_limit - self.risk_metrics['total_drawdown']
        
        if remaining_total_risk <= 0:
            logger.warning("⚠️ Total drawdown limit reached - trading suspended")
            return 0.0
        
        # Portfolio risk constraint
        current_portfolio_risk = self._calculate_current_portfolio_risk()
        max_portfolio_risk = self.initial_balance * (self.risk_parameters['max_portfolio_risk_percent'] / 100)
        remaining_portfolio_risk = max_portfolio_risk - current_portfolio_risk
        
        # Adjust position size based on remaining risk
        risk_adjusted_size = min(
            position_size,
            remaining_daily_risk / self.current_balance,
            remaining_total_risk / self.current_balance,
            remaining_portfolio_risk / self.current_balance
        )
        
        return max(0.0, risk_adjusted_size)
    
    def _calculate_trade_risk_metrics(self, position_size: float, entry_price: float, 
                                    stop_loss: float, symbol: str) -> Dict:
        """Calculate comprehensive risk metrics for a trade"""
        try:
            # Calculate potential loss
            price_risk = abs(entry_price - stop_loss)
            potential_loss = position_size * self.current_balance * price_risk / entry_price
            
            # Calculate potential reward (assuming 2:1 R:R)
            potential_reward = potential_loss * self.risk_parameters['target_risk_reward_ratio']
            
            # Risk as percentage of account
            risk_percent = (potential_loss / self.current_balance) * 100
            
            # Risk-reward ratio
            risk_reward_ratio = potential_reward / potential_loss if potential_loss > 0 else 0
            
            # Value at Risk (simplified)
            var_95 = self._calculate_var_95(position_size, symbol)
            
            return {
                'risk_percent': risk_percent,
                'potential_loss': potential_loss,
                'potential_reward': potential_reward,
                'risk_reward_ratio': risk_reward_ratio,
                'var_95': var_95,
                'position_value': position_size * self.current_balance
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {
                'risk_percent': 0.0,
                'potential_loss': 0.0,
                'potential_reward': 0.0,
                'risk_reward_ratio': 0.0,
                'var_95': 0.0,
                'position_value': 0.0
            }
    
    def _calculate_var_95(self, position_size: float, symbol: str) -> float:
        """Calculate Value at Risk at 95% confidence level"""
        try:
            # Simplified VaR calculation
            volatility = self._get_current_volatility(symbol) or 0.01
            position_value = position_size * self.current_balance
            
            # 1-day VaR at 95% confidence
            var_95 = position_value * volatility * norm.ppf(0.95)
            
            return var_95
        except:
            return 0.0
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol"""
        # This would integrate with your market data
        # For now, return a default value
        return 0.01  # 1% volatility
    
    def _calculate_current_portfolio_risk(self) -> float:
        """Calculate current portfolio risk"""
        total_risk = 0.0
        for position in self.open_positions.values():
            total_risk += position.get('risk_amount', 0)
        return total_risk
    
    def _get_fallback_position_size(self) -> Dict:
        """Fallback position sizing when calculations fail"""
        return {
            'optimal_size': 0.01,  # 1% fallback
            'risk_percent': 1.0,
            'potential_loss': self.current_balance * 0.01,
            'potential_reward': self.current_balance * 0.02,
            'risk_reward_ratio': 2.0,
            'methodology_breakdown': {
                'kelly_size': 0.01,
                'volatility_size': 0.01,
                'risk_based_size': 0.01,
                'combined_size': 0.01
            },
            'risk_metrics': {
                'risk_percent': 1.0,
                'potential_loss': self.current_balance * 0.01,
                'potential_reward': self.current_balance * 0.02,
                'risk_reward_ratio': 2.0,
                'var_95': self.current_balance * 0.015,
                'position_value': self.current_balance * 0.01
            }
        }
    
    def update_risk_metrics(self, new_balance: float, new_positions: Dict):
        """Update risk metrics with new balance and positions"""
        self.current_balance = new_balance
        self.open_positions = new_positions
        
        # Update daily PnL
        self.risk_metrics['daily_pnl'] = new_balance - self.risk_metrics.get('daily_start_balance', new_balance)
        
        # Update drawdown calculations
        if new_balance > self.risk_metrics['peak_balance']:
            self.risk_metrics['peak_balance'] = new_balance
            self.risk_metrics['current_drawdown'] = 0.0
        else:
            self.risk_metrics['current_drawdown'] = self.risk_metrics['peak_balance'] - new_balance
        
        self.risk_metrics['total_drawdown'] = self.initial_balance - new_balance
        
        # Update portfolio risk metrics
        self._update_portfolio_risk_metrics()
    
    def _update_portfolio_risk_metrics(self):
        """Update comprehensive portfolio risk metrics"""
        # Calculate VaR for entire portfolio
        portfolio_var = self._calculate_portfolio_var()
        self.risk_metrics['var_95'] = portfolio_var
        
        # Calculate Expected Shortfall (simplified)
        self.risk_metrics['expected_shortfall'] = portfolio_var * 1.3  # ES is typically higher than VaR
        
        # Update performance ratios (would require historical data)
        # self.risk_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio()
        # self.risk_metrics['calmar_ratio'] = self._calculate_calmar_ratio()
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.risk_metrics['daily_pnl'],
            'daily_pnl_percent': (self.risk_metrics['daily_pnl'] / self.initial_balance) * 100,
            'total_drawdown': self.risk_metrics['total_drawdown'],
            'total_drawdown_percent': (self.risk_metrics['total_drawdown'] / self.initial_balance) * 100,
            'current_drawdown': self.risk_metrics['current_drawdown'],
            'current_drawdown_percent': (self.risk_metrics['current_drawdown'] / self.risk_metrics['peak_balance']) * 100,
            'var_95': self.risk_metrics['var_95'],
            'expected_shortfall': self.risk_metrics['expected_shortfall'],
            'open_positions_count': len(self.open_positions),
            'portfolio_risk': self._calculate_current_portfolio_risk(),
            'remaining_daily_risk': self._calculate_remaining_daily_risk(),
            'remaining_total_risk': self._calculate_remaining_total_risk(),
            'ftmo_compliance': self._check_ftmo_compliance()
        }
    
    def _calculate_remaining_daily_risk(self) -> float:
        """Calculate remaining daily risk capacity"""
        daily_limit = self.initial_balance * (self.risk_parameters['max_daily_loss_percent'] / 100)
        return max(0, daily_limit - self.risk_metrics['daily_pnl'])
    
    def _calculate_remaining_total_risk(self) -> float:
        """Calculate remaining total risk capacity"""
        total_limit = self.initial_balance * (self.risk_parameters['max_total_drawdown_percent'] / 100)
        return max(0, total_limit - self.risk_metrics['total_drawdown'])
    
    def _check_ftmo_compliance(self) -> Dict:
        """Check FTMO rule compliance"""
        daily_ok = self.risk_metrics['daily_pnl'] >= -self._calculate_remaining_daily_risk()
        total_ok = self.risk_metrics['total_drawdown'] <= self.risk_parameters['max_total_drawdown_percent']
        
        return {
            'daily_loss_rule': daily_ok,
            'total_drawdown_rule': total_ok,
            'overall_compliance': daily_ok and total_ok,
            'trading_allowed': daily_ok and total_ok
        }
    
    def _calculate_portfolio_var(self) -> float:
        """Calculate portfolio Value at Risk"""
        # Simplified portfolio VaR calculation
        total_position_value = sum(pos.get('position_value', 0) for pos in self.open_positions.values())
        avg_volatility = 0.01  # Average volatility
        
        return total_position_value * avg_volatility * norm.ppf(0.95)

class AdvancedPositionOptimizer:
    """Advanced position optimization using modern portfolio theory"""
    
    def __init__(self, risk_manager: ProfessionalRiskManager):
        self.risk_manager = risk_manager
        self.optimization_methods = ['kelly', 'volatility', 'risk_based', 'sharpe_optimized']
    
    def optimize_portfolio_allocation(self, trading_signals: Dict) -> Dict:
        """Optimize portfolio allocation across multiple symbols"""
        try:
            # Calculate individual position sizes
            position_sizes = {}
            for symbol, signal in trading_signals.items():
                if signal.get('action') != 'hold' and signal.get('confidence', 0) > 0.3:
                    size_info = self.risk_manager.calculate_optimal_position_size(
                        symbol=symbol,
                        entry_price=signal.get('price', 0),
                        stop_loss=signal.get('stop_loss', 0),
                        confidence=signal.get('confidence', 0),
                        market_regime=signal.get('market_regime', 2),
                        portfolio_correlation=0.0  # Would calculate actual correlation
                    )
                    position_sizes[symbol] = size_info
            
            # Apply portfolio constraints
            optimized_sizes = self._apply_portfolio_constraints(position_sizes)
            
            return optimized_sizes
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {}
    
    def _apply_portfolio_constraints(self, position_sizes: Dict) -> Dict:
        """Apply portfolio-level constraints to position sizes"""
        total_allocated = sum(size_info.get('optimal_size', 0) for size_info in position_sizes.values())
        max_portfolio_allocation = 0.3  # Maximum 30% of portfolio in positions
        
        if total_allocated <= max_portfolio_allocation:
            return position_sizes
        
        # Scale down positions proportionally
        scale_factor = max_portfolio_allocation / total_allocated
        
        scaled_sizes = {}
        for symbol, size_info in position_sizes.items():
            scaled_size = size_info.copy()
            scaled_size['optimal_size'] *= scale_factor
            scaled_sizes[symbol] = scaled_size
        
        logger.info(f"📊 Portfolio allocation scaled down by factor {scale_factor:.2f}")
        return scaled_sizes
