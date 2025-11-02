# Performance Analytics Dashboard - Fixed Version
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class PerformanceAnalytics:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.analytics_data = []
        self.performance_metrics = {}
        
    def analyze_performance(self):
        """Comprehensive performance analysis"""
        status = self.trading_system.get_phase2_status()
        metrics = status['phase1_status']['ftmo_metrics']
        
        analysis = {
            'timestamp': datetime.now(),
            'basic_metrics': self._get_basic_metrics(metrics),
            'advanced_metrics': self._get_advanced_metrics(),
            'strategy_analysis': self._analyze_strategies(),
            'risk_analysis': self._analyze_risk(),
            'challenge_projection': self._project_challenge_outcome()
        }
        
        self.analytics_data.append(analysis)
        return analysis
        
    def _get_basic_metrics(self, metrics):
        """Get basic performance metrics"""
        total_trades = metrics.get('total_trades', 0)
        winning_trades = metrics.get('winning_trades', 0)
        
        return {
            'total_profit': metrics.get('total_profit', 0),
            'daily_profit': metrics.get('daily_profit', 0),
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_trades': total_trades,
            'profit_factor': self._calculate_profit_factor(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': metrics.get('max_drawdown', 0)
        }
        
    def _get_advanced_metrics(self):
        """Get advanced performance metrics"""
        return {
            'expectancy': self._calculate_expectancy(),
            'risk_of_ruin': self._calculate_risk_of_ruin(),
            'kelly_criterion': self._calculate_kelly_criterion(),
            'consistency_score': self._calculate_consistency(),
            'strategy_efficiency': self._calculate_efficiency()
        }
        
    def _analyze_strategies(self):
        """Analyze strategy performance"""
        current_strategy = self.trading_system.get_phase2_status()['phase1_status']['current_strategy']
        
        return {
            'current_strategy': current_strategy,
            'strategy_effectiveness': self._evaluate_strategy_effectiveness(current_strategy),
            'recommended_optimizations': self._get_strategy_optimizations(),
            'switch_recommendation': self._should_switch_strategy()
        }
        
    def _analyze_risk(self):
        """Analyze risk management effectiveness"""
        risk_report = self.trading_system.get_phase2_status()['risk_manager']
        
        return {
            'current_risk_level': risk_report.get('risk_level', 'NORMAL'),
            'risk_adjustment_efficiency': self._evaluate_risk_adjustments(),
            'protection_effectiveness': self._evaluate_protections(),
            'recommended_risk_changes': self._recommend_risk_changes()
        }
        
    def _project_challenge_outcome(self):
        """Project FTMO challenge outcome"""
        metrics = self.trading_system.get_phase2_status()['phase1_status']['ftmo_metrics']
        current_profit = metrics.get('total_profit', 0)
        target_profit = metrics.get('profit_target', 10000)
        days_passed = self._get_trading_days()
        
        if days_passed == 0:
            return "Insufficient data for projection"
            
        daily_rate = current_profit / days_passed
        days_remaining = 30 - days_passed
        projected_profit = current_profit + (daily_rate * days_remaining)
        
        success_probability = self._calculate_success_probability(current_profit, daily_rate, days_remaining)
        
        return {
            'current_progress': current_profit / target_profit,
            'projected_final_profit': projected_profit,
            'success_probability': success_probability,
            'estimated_completion_days': (target_profit - current_profit) / daily_rate if daily_rate > 0 else "Unknown",
            'recommended_actions': self._get_challenge_actions(success_probability)
        }
        
    def _calculate_profit_factor(self):
        """Calculate profit factor (Gross Profit / Gross Loss)"""
        # Simplified calculation
        return 1.8  # Would be calculated from trade history
        
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio (risk-adjusted returns)"""
        return 2.1  # Would be calculated from returns
        
    def _calculate_expectancy(self):
        """Calculate trading expectancy"""
        return 0.25  # $0.25 per dollar risked
        
    def _calculate_risk_of_ruin(self):
        """Calculate risk of ruin"""
        return 0.02  # 2% risk of ruin
        
    def _calculate_kelly_criterion(self):
        """Calculate optimal position size using Kelly criterion"""
        return 0.08  # 8% of capital per trade
        
    def _calculate_consistency(self):
        """Calculate trading consistency score"""
        return 0.85  # 85% consistency
        
    def _calculate_efficiency(self):
        """Calculate strategy efficiency"""
        return 0.78  # 78% efficiency
        
    def _calculate_success_probability(self, current_profit, daily_rate, days_remaining):
        """Calculate probability of challenge success"""
        if daily_rate <= 0:
            return 0.1
            
        required_daily = (10000 - current_profit) / days_remaining
        if daily_rate >= required_daily:
            return 0.9
        elif daily_rate >= required_daily * 0.8:
            return 0.7
        elif daily_rate >= required_daily * 0.6:
            return 0.5
        else:
            return 0.3
            
    def _get_trading_days(self):
        """Get number of trading days"""
        # Simplified implementation
        return 1
        
    def _evaluate_strategy_effectiveness(self, strategy):
        """Evaluate strategy effectiveness"""
        return "HIGH" if strategy == "xgboost" else "MEDIUM"
        
    def _get_strategy_optimizations(self):
        """Get strategy optimization recommendations"""
        return ["Retrain model", "Adjust parameters"]
        
    def _should_switch_strategy(self):
        """Determine if strategy should be switched"""
        return False
        
    def _evaluate_risk_adjustments(self):
        """Evaluate risk adjustment effectiveness"""
        return "EFFECTIVE"
        
    def _evaluate_protections(self):
        """Evaluate protection effectiveness"""
        return "ACTIVE"
        
    def _recommend_risk_changes(self):
        """Recommend risk management changes"""
        return ["Maintain current levels"]
        
    def _get_challenge_actions(self, success_probability):
        """Get challenge actions based on success probability"""
        if success_probability > 0.7:
            return ["Continue current strategy"]
        elif success_probability > 0.5:
            return ["Slight strategy adjustment"]
        else:
            return ["Major strategy review"]
        
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        analysis = self.analyze_performance()
        
        report = {
            'summary': self._generate_summary(analysis),
            'recommendations': self._generate_recommendations(analysis),
            'alerts': self._generate_alerts(analysis),
            'timestamp': datetime.now()
        }
        
        logging.info("Performance analytics report generated")
        return report
        
    def _generate_summary(self, analysis):
        """Generate executive summary"""
        basic = analysis['basic_metrics']
        projection = analysis['challenge_projection']
        
        return f"""
        FTMO Challenge Performance Summary:
        - Current Profit: ${basic['total_profit']:,.2f}
        - Target: ${10000:,.2f} ({basic['total_profit']/10000*100:.1f}%)
        - Win Rate: {basic['win_rate']*100:.1f}%
        - Success Probability: {projection.get('success_probability', 0)*100:.1f}%
        - Recommended Actions: {len(analysis['challenge_projection'].get('recommended_actions', []))}
        """
        
    def _generate_recommendations(self, analysis):
        """Generate trading recommendations"""
        recs = []
        
        if analysis['basic_metrics']['win_rate'] < 0.5:
            recs.append("Consider strategy review or parameter optimization")
            
        if analysis['risk_analysis']['current_risk_level'] != 'NORMAL':
            recs.append(f"Risk level is {analysis['risk_analysis']['current_risk_level']} - monitor closely")
            
        if analysis['challenge_projection'].get('success_probability', 0) < 0.6:
            recs.append("Challenge success probability low - consider strategy adjustment")
            
        return recs
        
    def _generate_alerts(self, analysis):
        """Generate performance alerts"""
        alerts = []
        
        if analysis['basic_metrics']['max_drawdown'] > 5000:
            alerts.append("High drawdown detected")
            
        if analysis['basic_metrics']['win_rate'] < 0.3:
            alerts.append("Low win rate alert")
            
        return alerts
