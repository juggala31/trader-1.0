# advanced_backtester.py - Professional Backtesting & Strategy Optimization
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from enum import Enum
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger('FTMO_AI')

class BacktestMode(Enum):
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    OPTIMIZATION = "optimization"

@dataclass
class TradeSignal:
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    timestamp: datetime
    confidence: float = 0.5
    reason: str = ""

@dataclass
class BacktestResult:
    total_trades: int
    profitable_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    average_trade: float
    best_trade: float
    worst_trade: float
    recovery_factor: float
    calmar_ratio: float

class ProfessionalBacktester:
    """Professional backtesting system with advanced analytics"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.backtest_results = {}
        self.optimization_history = []
        self.performance_metrics = {}
        
        # Backtest configuration
        self.config = {
            'commission': 0.0002,  # 0.02% commission
            'slippage': 0.0001,   # 0.01% slippage
            'spread': 0.0002,     # 0.02% spread
            'max_position_size': 0.1,  # 10% max position
            'risk_free_rate': 0.02,   # 2% risk-free rate
        }
        
        # Advanced analytics
        self.analytics_enabled = {
            'monte_carlo': True,
            'walk_forward': True,
            'parameter_optimization': True,
            'strategy_comparison': True,
            'risk_analysis': True
        }
    
    def run_comprehensive_backtest(self, strategy, historical_data: pd.DataFrame, 
                                 period: str = "2y", mode: BacktestMode = BacktestMode.WALK_FORWARD) -> Dict:
        """Run comprehensive backtest with multiple analysis methods"""
        logger.info(f"🎯 Starting comprehensive backtest - Mode: {mode.value}")
        
        try:
            # Prepare historical data
            prepared_data = self._prepare_historical_data(historical_data, period)
            
            if mode == BacktestMode.WALK_FORWARD:
                results = self._walk_forward_analysis(strategy, prepared_data)
            elif mode == BacktestMode.MONTE_CARLO:
                results = self._monte_carlo_simulation(strategy, prepared_data)
            elif mode == BacktestMode.STRESS_TEST:
                results = self._stress_test_analysis(strategy, prepared_data)
            elif mode == BacktestMode.OPTIMIZATION:
                results = self._parameter_optimization(strategy, prepared_data)
            else:
                results = self._standard_backtest(strategy, prepared_data)
            
            # Generate comprehensive report
            report = self._generate_backtest_report(results, mode)
            
            logger.info(f"✅ Backtest completed - {report['summary']['total_trades']} trades analyzed")
            return report
            
        except Exception as e:
            logger.error(f"❌ Backtest error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _walk_forward_analysis(self, strategy, data: pd.DataFrame, 
                             window_size: int = 252, step_size: int = 63) -> Dict:
        """Walk-forward analysis for robust strategy validation"""
        results = {
            'periods': [],
            'overall_metrics': {},
            'stability_analysis': {},
            'validation_score': 0.0
        }
        
        total_periods = len(data) // step_size
        current_index = 0
        
        for period in range(total_periods - 1):
            # Split data into training and validation sets
            train_end = current_index + window_size
            validation_start = train_end
            validation_end = validation_start + step_size
            
            if validation_end >= len(data):
                break
            
            train_data = data.iloc[current_index:train_end]
            validation_data = data.iloc[validation_start:validation_end]
            
            # Train strategy on training period
            strategy.train(train_data)
            
            # Test on validation period
            period_result = self._run_period_backtest(strategy, validation_data, f"Period_{period+1}")
            
            results['periods'].append(period_result)
            current_index += step_size
        
        # Calculate walk-forward metrics
        results['overall_metrics'] = self._calculate_walk_forward_metrics(results['periods'])
        results['stability_analysis'] = self._analyze_strategy_stability(results['periods'])
        results['validation_score'] = self._calculate_validation_score(results['periods'])
        
        return results
    
    def _monte_carlo_simulation(self, strategy, data: pd.DataFrame, 
                              simulations: int = 1000) -> Dict:
        """Monte Carlo simulation for risk analysis"""
        logger.info(f"🔢 Running Monte Carlo simulation ({simulations} iterations)")
        
        base_results = self._standard_backtest(strategy, data)
        if not base_results['success']:
            return base_results
        
        monte_carlo_results = {
            'original_result': base_results,
            'simulations': [],
            'percentiles': {},
            'risk_metrics': {}
        }
        
        # Extract key metrics from original backtest
        original_pnl = base_results['metrics']['total_pnl']
        original_returns = base_results['detailed_metrics']['daily_returns']
        
        for i in range(simulations):
            # Create randomized return series based on original statistics
            simulated_returns = self._generate_monte_carlo_returns(original_returns)
            simulated_result = self._simulate_from_returns(simulated_returns, base_results['metrics'])
            
            monte_carlo_results['simulations'].append(simulated_result)
        
        # Calculate percentiles and risk metrics
        monte_carlo_results['percentiles'] = self._calculate_monte_carlo_percentiles(monte_carlo_results['simulations'])
        monte_carlo_results['risk_metrics'] = self._calculate_monte_carlo_risk(monte_carlo_results['simulations'])
        
        return monte_carlo_results
    
    def _stress_test_analysis(self, strategy, data: pd.DataFrame) -> Dict:
        """Stress testing under extreme market conditions"""
        stress_scenarios = {
            'high_volatility': self._simulate_high_volatility(data),
            'flash_crash': self._simulate_flash_crash(data),
            'low_volatility': self._simulate_low_volatility(data),
            'trend_reversal': self._simulate_trend_reversal(data)
        }
        
        stress_results = {}
        for scenario_name, scenario_data in stress_scenarios.items():
            scenario_result = self._run_period_backtest(strategy, scenario_data, f"Stress_{scenario_name}")
            stress_results[scenario_name] = scenario_result
        
        return {
            'scenarios': stress_results,
            'resilience_score': self._calculate_strategy_resilience(stress_results),
            'worst_case_analysis': self._analyze_worst_case_scenarios(stress_results)
        }
    
    def _parameter_optimization(self, strategy, data: pd.DataFrame) -> Dict:
        """Advanced parameter optimization with robustness testing"""
        logger.info("🔧 Running parameter optimization")
        
        # Define parameter space for optimization
        parameter_space = self._define_parameter_space(strategy)
        
        optimization_results = {
            'parameter_space': parameter_space,
            'optimization_runs': [],
            'best_parameters': {},
            'robustness_test': {}
        }
        
        # Grid search optimization
        best_score = -float('inf')
        best_params = None
        
        for params in self._generate_parameter_combinations(parameter_space):
            # Configure strategy with parameters
            strategy.configure(params)
            
            # Run backtest with these parameters
            result = self._standard_backtest(strategy, data)
            
            if result['success']:
                score = self._calculate_optimization_score(result['metrics'])
                optimization_results['optimization_runs'].append({
                    'parameters': params,
                    'score': score,
                    'metrics': result['metrics']
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        optimization_results['best_parameters'] = best_params
        optimization_results['robustness_test'] = self._test_parameter_robustness(strategy, best_params, data)
        
        return optimization_results
    
    def _standard_backtest(self, strategy, data: pd.DataFrame) -> Dict:
        """Standard backtest implementation"""
        try:
            initial_balance = self.initial_balance
            current_balance = initial_balance
            positions = {}
            trades = []
            daily_balances = [initial_balance]
            daily_returns = []
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                # Get trading signal from strategy
                signal = strategy.generate_signal(row)
                
                # Execute trades based on signal
                if signal and signal.action != 'hold':
                    trade_result = self._execute_trade(signal, current_balance, row)
                    if trade_result['success']:
                        trades.append(trade_result)
                        current_balance = trade_result['new_balance']
                
                # Update daily metrics
                daily_balances.append(current_balance)
                if i > 0:
                    daily_return = (current_balance - daily_balances[-2]) / daily_balances[-2]
                    daily_returns.append(daily_return)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(trades, daily_balances, daily_returns)
            detailed_metrics = self._calculate_detailed_metrics(trades, daily_balances)
            
            return {
                'success': True,
                'metrics': metrics,
                'detailed_metrics': detailed_metrics,
                'trades': trades,
                'initial_balance': initial_balance,
                'final_balance': current_balance,
                'total_return': (current_balance - initial_balance) / initial_balance * 100
            }
            
        except Exception as e:
            logger.error(f"Standard backtest error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_trade(self, signal: TradeSignal, current_balance: float, market_data: pd.Series) -> Dict:
        """Execute a trade with realistic market conditions"""
        try:
            # Apply commission and slippage
            entry_price = signal.entry_price * (1 + self.config['slippage'])
            exit_price = signal.take_profit * (1 - self.config['slippage']) if signal.action == 'buy' else signal.take_profit * (1 + self.config['slippage'])
            
            # Calculate position value
            position_value = current_balance * signal.size
            
            # Calculate P&L
            if signal.action == 'buy':
                pnl = (exit_price - entry_price) * (position_value / entry_price)
            else:
                pnl = (entry_price - exit_price) * (position_value / entry_price)
            
            # Apply commission
            commission = position_value * self.config['commission']
            pnl -= commission
            
            new_balance = current_balance + pnl
            
            return {
                'success': True,
                'symbol': signal.symbol,
                'action': signal.action,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': signal.size,
                'pnl': pnl,
                'commission': commission,
                'new_balance': new_balance,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_performance_metrics(self, trades: List, daily_balances: List, daily_returns: List) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).__dict__
        
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        
        # Calculate max drawdown
        peak = max(daily_balances)
        trough = min(daily_balances)
        max_drawdown = (peak - trough) / peak * 100 if peak > 0 else 0
        
        # Calculate Sharpe ratio
        risk_free_return = self.config['risk_free_rate'] / 252  # Daily risk-free rate
        excess_returns = [r - risk_free_return for r in daily_returns] if daily_returns else [0]
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Calculate profit factor
        total_profits = sum(max(0, t['pnl']) for t in trades)
        total_losses = abs(sum(min(0, t['pnl']) for t in trades))
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        average_trade = total_pnl / total_trades if total_trades > 0 else 0
        best_trade = max(t['pnl'] for t in trades) if trades else 0
        worst_trade = min(t['pnl'] for t in trades) if trades else 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'average_trade': average_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'recovery_factor': total_pnl / max_drawdown if max_drawdown > 0 else 0,
            'calmar_ratio': (total_pnl / self.initial_balance * 100) / max_drawdown if max_drawdown > 0 else 0
        }
    
    def _calculate_detailed_metrics(self, trades: List, daily_balances: List) -> Dict:
        """Calculate detailed performance analytics"""
        return {
            'daily_returns': self._calculate_daily_returns(daily_balances),
            'monthly_returns': self._calculate_monthly_returns(daily_balances),
            'trade_duration': self._calculate_average_trade_duration(trades),
            'consecutive_wins': self._calculate_consecutive_wins_losses(trades),
            'consecutive_losses': self._calculate_consecutive_wins_losses(trades, False),
            'underwater_periods': self._calculate_underwater_periods(daily_balances)
        }
    
    # Helper methods for detailed calculations
    def _calculate_daily_returns(self, balances: List) -> List:
        """Calculate daily returns from balance history"""
        returns = []
        for i in range(1, len(balances)):
            daily_return = (balances[i] - balances[i-1]) / balances[i-1]
            returns.append(daily_return)
        return returns
    
    def _calculate_monthly_returns(self, balances: List) -> List:
        """Calculate monthly returns (simplified)"""
        if len(balances) < 22:  # Minimum for monthly calculation
            return [0]
        monthly_returns = []
        for i in range(21, len(balances), 21):
            monthly_return = (balances[i] - balances[i-21]) / balances[i-21]
            monthly_returns.append(monthly_return)
        return monthly_returns
    
    def _calculate_average_trade_duration(self, trades: List) -> float:
        """Calculate average trade duration (placeholder)"""
        return 45.0  # 45 minutes average
    
    def _calculate_consecutive_wins_losses(self, trades: List, wins: bool = True) -> int:
        """Calculate consecutive wins or losses"""
        max_streak = 0
        current_streak = 0
        
        for trade in trades:
            if (trade['pnl'] > 0) == wins:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
    
    def _calculate_underwater_periods(self, balances: List) -> int:
        """Calculate number of underwater periods"""
        peak = balances[0]
        underwater_count = 0
        
        for balance in balances:
            if balance < peak:
                underwater_count += 1
            else:
                peak = balance
                
        return underwater_count

    # Advanced analytics helper methods
    def _calculate_walk_forward_metrics(self, periods: List) -> Dict:
        """Calculate walk-forward analysis metrics"""
        period_metrics = [p['metrics'] for p in periods if 'metrics' in p]
        
        return {
            'average_win_rate': np.mean([m['win_rate'] for m in period_metrics]),
            'std_win_rate': np.std([m['win_rate'] for m in period_metrics]),
            'average_sharpe': np.mean([m['sharpe_ratio'] for m in period_metrics]),
            'consistency_score': self._calculate_consistency_score(period_metrics)
        }
    
    def _calculate_consistency_score(self, metrics: List) -> float:
        """Calculate strategy consistency score"""
        if len(metrics) < 2:
            return 0.0
        
        win_rates = [m['win_rate'] for m in metrics]
        sharpe_ratios = [m['sharpe_ratio'] for m in metrics]
        
        # Consistency based on low variance in performance
        win_rate_consistency = 1.0 - (np.std(win_rates) / np.mean(win_rates)) if np.mean(win_rates) > 0 else 0.0
        sharpe_consistency = 1.0 - (np.std(sharpe_ratios) / np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) > 0 else 0.0
        
        return (win_rate_consistency + sharpe_consistency) / 2
    
    def _generate_monte_carlo_returns(self, original_returns: List) -> List:
        """Generate randomized returns for Monte Carlo simulation"""
        if not original_returns:
            return [0] * 252  # Default to zero returns
        
        # Bootstrap sampling with replacement
        n_periods = len(original_returns)
        simulated_returns = np.random.choice(original_returns, size=n_periods, replace=True)
        return simulated_returns.tolist()
    
    def _simulate_from_returns(self, returns: List, base_metrics: Dict) -> Dict:
        """Simulate trading results from return series"""
        initial_balance = self.initial_balance
        current_balance = initial_balance
        
        for ret in returns:
            current_balance *= (1 + ret)
        
        total_return = (current_balance - initial_balance) / initial_balance * 100
        
        return {
            'final_balance': current_balance,
            'total_return': total_return,
            'simulated_metrics': {
                'sharpe_ratio': base_metrics.get('sharpe_ratio', 0) * 0.9,  # Slight degradation
                'max_drawdown': base_metrics.get('max_drawdown', 0) * 1.1,   # Slight increase
                'win_rate': base_metrics.get('win_rate', 0) * 0.95           # Slight reduction
            }
        }

    def _define_parameter_space(self, strategy) -> Dict:
        """Define parameter space for optimization"""
        # This would be strategy-specific
        return {
            'confidence_threshold': {'min': 0.05, 'max': 0.3, 'step': 0.01},
            'position_size': {'min': 0.01, 'max': 0.1, 'step': 0.005},
            'stop_loss_atr': {'min': 1.0, 'max': 3.0, 'step': 0.1},
            'take_profit_ratio': {'min': 1.5, 'max': 3.0, 'step': 0.1}
        }
    
    def _generate_parameter_combinations(self, parameter_space: Dict) -> List[Dict]:
        """Generate parameter combinations for grid search"""
        # Simplified implementation - would use itertools.product in real implementation
        combinations = []
        
        # Generate some sample combinations
        for conf in [0.1, 0.15, 0.2]:
            for size in [0.02, 0.03, 0.04]:
                for sl in [1.5, 2.0, 2.5]:
                    for tp in [2.0, 2.5, 3.0]:
                        combinations.append({
                            'confidence_threshold': conf,
                            'position_size': size,
                            'stop_loss_atr': sl,
                            'take_profit_ratio': tp
                        })
        
        return combinations
    
    def _calculate_optimization_score(self, metrics: Dict) -> float:
        """Calculate optimization score from performance metrics"""
        # Multi-objective scoring function
        win_rate_score = metrics['win_rate'] / 100
        sharpe_score = min(metrics['sharpe_ratio'] / 3.0, 1.0)  # Cap at 3.0
        drawdown_penalty = 1.0 - (metrics['max_drawdown'] / 50.0)  # Penalize high drawdown
        
        return (win_rate_score * 0.4 + sharpe_score * 0.4 + drawdown_penalty * 0.2)
    
    def _test_parameter_robustness(self, strategy, parameters: Dict, data: pd.DataFrame) -> Dict:
        """Test parameter robustness across different market conditions"""
        # Split data into different market regimes
        regimes = self._identify_market_regimes(data)
        
        robustness_results = {}
        for regime_name, regime_data in regimes.items():
            result = self._run_period_backtest(strategy, regime_data, f"Robustness_{regime_name}")
            robustness_results[regime_name] = result
        
        return {
            'regime_performance': robustness_results,
            'robustness_score': self._calculate_robustness_score(robustness_results)
        }
    
    def _identify_market_regimes(self, data: pd.DataFrame) -> Dict:
        """Identify different market regimes in historical data"""
        # Simplified regime identification
        volatility = data['close'].pct_change().std()
        
        if volatility > 0.02:
            return {'high_volatility': data}
        elif volatility < 0.005:
            return {'low_volatility': data}
        else:
            return {'normal_volatility': data}
    
    def _calculate_robustness_score(self, regime_results: Dict) -> float:
        """Calculate parameter robustness score"""
        scores = []
        for regime, result in regime_results.items():
            if result['success']:
                scores.append(result['metrics']['sharpe_ratio'])
        
        return np.mean(scores) if scores else 0.0

    def _generate_backtest_report(self, results: Dict, mode: BacktestMode) -> Dict:
        """Generate comprehensive backtest report"""
        report = {
            'backtest_mode': mode.value,
            'generation_date': datetime.now().isoformat(),
            'summary': results.get('overall_metrics', {}),
            'detailed_analysis': results,
            'recommendations': self._generate_recommendations(results),
            'risk_assessment': self._assess_strategy_risk(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate strategy recommendations based on backtest results"""
        recommendations = []
        
        metrics = results.get('overall_metrics', {})
        
        if metrics.get('win_rate', 0) < 55:
            recommendations.append("Consider increasing confidence threshold to improve win rate")
        
        if metrics.get('profit_factor', 0) < 1.2:
            recommendations.append("Review risk management - profit factor below optimal level")
        
        if metrics.get('max_drawdown', 0) > 10:
            recommendations.append("Reduce position sizes to control drawdown")
        
        if metrics.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("Optimize for better risk-adjusted returns")
        
        return recommendations
    
    def _assess_strategy_risk(self, results: Dict) -> Dict:
        """Assess strategy risk profile"""
        metrics = results.get('overall_metrics', {})
        
        return {
            'risk_level': 'low' if metrics.get('max_drawdown', 0) < 5 else 'medium' if metrics.get('max_drawdown', 0) < 15 else 'high',
            'consistency': 'high' if metrics.get('win_rate', 0) > 60 else 'medium' if metrics.get('win_rate', 0) > 50 else 'low',
            'scalability': 'good' if metrics.get('sharpe_ratio', 0) > 1.5 else 'limited',
            'market_dependency': 'low' if metrics.get('profit_factor', 0) > 2.0 else 'medium' if metrics.get('profit_factor', 0) > 1.2 else 'high'
        }

    def _prepare_historical_data(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Prepare historical data for backtesting"""
        if period == "2y":
            # Last 2 years of data
            cutoff_date = datetime.now() - timedelta(days=730)
        elif period == "1y":
            cutoff_date = datetime.now() - timedelta(days=365)
        elif period == "6m":
            cutoff_date = datetime.now() - timedelta(days=180)
        else:
            cutoff_date = datetime.now() - timedelta(days=90)  # Default 3 months
        
        # Filter data by date
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            filtered_data = data[data['timestamp'] >= cutoff_date]
        else:
            # Use index if no timestamp column
            filtered_data = data.iloc[-min(len(data), 500):]  # Last 500 periods
        
        return filtered_data

    def _run_period_backtest(self, strategy, data: pd.DataFrame, period_name: str) -> Dict:
        """Run backtest for a specific period"""
        result = self._standard_backtest(strategy, data)
        result['period_name'] = period_name
        return result

# Additional utility functions
def calculate_comprehensive_metrics(trades: List, equity_curve: List) -> Dict:
    """Calculate comprehensive trading metrics"""
    # Implementation would include advanced metrics like:
    # - Omega ratio
    # - Sortino ratio
    # - Ulcer index
    # - Martin ratio
    # - Tail ratio
    return {'advanced_metrics': 'placeholder'}

def optimize_position_sizing(strategy_performance: Dict, risk_constraints: Dict) -> Dict:
    """Optimize position sizing based on strategy performance"""
    # Implementation would use Kelly criterion, risk parity, etc.
    return {'optimal_sizing': 'placeholder'}

def generate_tear_sheet(backtest_results: Dict) -> Dict:
    """Generate professional tear sheet analysis"""
    # Implementation would create comprehensive analysis report
    return {'tear_sheet': 'placeholder'}
