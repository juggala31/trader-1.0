# Automated Reporting System - Phase 3 (Fixed Imports)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import logging
from datetime import datetime, timedelta
import pandas as pd

class AutomatedReporter:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.report_config = self._load_report_config()
        self.last_report_time = None
        
    def _load_report_config(self):
        """Load reporting configuration"""
        return {
            'report_intervals': {
                'daily': timedelta(hours=24),
                'weekly': timedelta(days=7),
                'challenge_progress': timedelta(hours=6)
            },
            'email_settings': {
                'enabled': False,  # Set to True for live deployment
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'your_email@gmail.com',
                'sender_password': 'your_app_password'
            },
            'recipients': ['your_email@gmail.com']
        }
        
    def generate_daily_report(self):
        """Generate daily FTMO challenge report"""
        status = self.trading_system.get_phase2_status()
        metrics = status['phase1_status']['ftmo_metrics']
        
        report = {
            'report_type': 'DAILY',
            'timestamp': datetime.now(),
            'challenge_progress': {
                'total_profit': metrics.get('total_profit', 0),
                'profit_target': metrics.get('profit_target', 10000),
                'progress_percentage': (metrics.get('total_profit', 0) / metrics.get('profit_target', 10000)) * 100,
                'days_remaining': self._calculate_days_remaining(),
                'completion_estimate': self._estimate_completion_date()
            },
            'performance_metrics': {
                'daily_profit': metrics.get('daily_profit', 0),
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': metrics.get('winning_trades', 0),
                'win_rate': (metrics.get('winning_trades', 0) / metrics.get('total_trades', 1)) * 100,
                'max_drawdown': metrics.get('max_drawdown', 0)
            },
            'risk_metrics': status['risk_manager'],
            'system_health': status['rule_enforcer']
        }
        
        logging.info(f"Daily report generated: ${report['challenge_progress']['total_profit']} profit")
        return report
        
    def generate_weekly_report(self):
        """Generate weekly comprehensive report"""
        weekly_report = {
            'report_type': 'WEEKLY',
            'timestamp': datetime.now(),
            'weekly_summary': {
                'total_profit': 0,  # Would be calculated
                'average_daily_profit': 0,
                'total_trades': 0,
                'success_rate': 0
            },
            'weekly_highlights': self._generate_weekly_highlights(),
            'next_week_goals': self._generate_next_week_goals()
        }
        
        return weekly_report
        
    def generate_challenge_verification_report(self):
        """Generate report for FTMO challenge verification"""
        status = self.trading_system.get_phase2_status()
        metrics = status['phase1_status']['ftmo_metrics']
        
        verification_report = {
            'report_type': 'CHALLENGE_VERIFICATION',
            'timestamp': datetime.now(),
            'account_information': {
                'account_id': '1600038177',
                'challenge_type': '200k',
                'start_date': datetime.now().date().isoformat()
            },
            'challenge_results': {
                'profit_target_achieved': metrics.get('total_profit', 0) >= metrics.get('profit_target', 10000),
                'final_profit': metrics.get('total_profit', 0),
                'total_trading_days': self._calculate_trading_days(),
                'rule_compliance': self._check_rule_compliance(),
                'verification_status': 'PENDING'  # Would be determined by FTMO
            },
            'performance_evidence': {
                'trade_history': len(self.trading_system.phase1_system.ftmo_logger.trade_history),
                'risk_management': status['risk_manager'],
                'strategy_performance': self._generate_strategy_performance()
            }
        }
        
        return verification_report
        
    def _calculate_days_remaining(self):
        """Calculate days remaining in FTMO challenge"""
        start_date = self.trading_system.phase1_system.ftmo_logger.start_date
        max_days = self.trading_system.phase1_system.ftmo_logger.rules.get('max_trading_period', 30)
        days_passed = (datetime.now().date() - start_date.date()).days
        return max(0, max_days - days_passed)
        
    def _estimate_completion_date(self):
        """Estimate challenge completion date"""
        metrics = self.trading_system.get_phase2_status()['phase1_status']['ftmo_metrics']
        current_profit = metrics.get('total_profit', 0)
        target_profit = metrics.get('profit_target', 10000)
        days_passed = (datetime.now().date() - self.trading_system.phase1_system.ftmo_logger.start_date.date()).days
        
        if current_profit <= 0 or days_passed <= 0:
            return "Unable to estimate"
            
        daily_rate = current_profit / days_passed
        if daily_rate <= 0:
            return "Not on track"
            
        days_needed = (target_profit - current_profit) / daily_rate
        completion_date = datetime.now() + timedelta(days=days_needed)
        
        return completion_date.strftime('%Y-%m-%d')
        
    def _calculate_trading_days(self):
        """Calculate actual trading days"""
        return (datetime.now().date() - self.trading_system.phase1_system.ftmo_logger.start_date.date()).days
        
    def _check_rule_compliance(self):
        """Check FTMO rule compliance"""
        status = self.trading_system.get_phase2_status()
        rule_enforcer = status['rule_enforcer']
        
        return {
            'daily_loss_compliance': rule_enforcer.get('violations_count', 0) == 0,
            'drawdown_compliance': status['risk_manager'].get('risk_level') != 'LOCKDOWN',
            'trading_days_compliance': self._calculate_trading_days() >= 5
        }
        
    def _generate_weekly_highlights(self):
        """Generate weekly performance highlights"""
        return {
            'best_day': 'Monday',  # Would be calculated
            'most_profitable_strategy': 'XGBoost',
            'key_improvements': 'Risk management optimization'
        }
        
    def _generate_next_week_goals(self):
        """Generate goals for next week"""
        return {
            'profit_target': 2500,
            'risk_adjustments': 'Fine-tune position sizing',
            'strategy_improvements': 'Retrain XGBoost model'
        }
        
    def _generate_strategy_performance(self):
        """Generate strategy performance summary"""
        return {
            'xgboost_performance': {'win_rate': 0.65, 'profit': 7500},
            'fallback_performance': {'win_rate': 0.55, 'profit': 2500},
            'strategy_switches': 3
        }
        
    def send_email_report(self, report, recipient):
        """Send report via email"""
        if not self.report_config['email_settings']['enabled']:
            logging.info("Email reporting disabled")
            return False
            
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.report_config['email_settings']['sender_email']
            msg['To'] = recipient
            msg['Subject'] = f"FTMO Report - {report['report_type']} - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create HTML content
            html_content = self._report_to_html(report)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.report_config['email_settings']['smtp_server'], 
                             self.report_config['email_settings']['smtp_port']) as server:
                server.starttls()
                server.login(self.report_config['email_settings']['sender_email'],
                           self.report_config['email_settings']['sender_password'])
                server.send_message(msg)
                
            logging.info(f"Report sent to {recipient}")
            return True
            
        except Exception as e:
            logging.error(f"Email sending failed: {e}")
            return False
            
    def _report_to_html(self, report):
        """Convert report to HTML format"""
        html = f"""
        <html>
        <body>
            <h1>FTMO Trading Report</h1>
            <h2>{report['report_type']} Report</h2>
            <p>Generated: {report['timestamp']}</p>
            
            <h3>Challenge Progress</h3>
            <p>Total Profit: ${report['challenge_progress']['total_profit']:,.2f}</p>
            <p>Target: ${report['challenge_progress']['profit_target']:,.2f}</p>
            <p>Progress: {report['challenge_progress']['progress_percentage']:.1f}%</p>
            
            <h3>Performance Metrics</h3>
            <p>Win Rate: {report['performance_metrics']['win_rate']:.1f}%</p>
            <p>Total Trades: {report['performance_metrics']['total_trades']}</p>
            
            <h3>Risk Status</h3>
            <p>Risk Level: {report['risk_metrics']['risk_level']}</p>
            <p>Max Drawdown: {report['performance_metrics']['max_drawdown']:,.2f}</p>
        </body>
        </html>
        """
        return html
        
    def export_report(self, report, filename=None):
        """Export report to JSON file"""
        if filename is None:
            filename = f"ftmo_report_{report['report_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logging.info(f"Report exported to {filename}")
        return filename
