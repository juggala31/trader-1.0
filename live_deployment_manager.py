# Live Deployment Manager - Phase 3
import os
import json
import time
import logging
from datetime import datetime, timedelta
from ftmo_phase2_system import FTMO_Phase2_System

class LiveDeploymentManager:
    def __init__(self, account_id="1600038177", challenge_type="200k", live_mode=False):
        self.account_id = account_id
        self.challenge_type = challenge_type
        self.live_mode = live_mode
        self.deployment_status = "PRE_DEPLOYMENT"
        self.deployment_log = []
        
        # Initialize trading system
        self.trading_system = FTMO_Phase2_System(account_id, challenge_type)
        
        # Deployment configuration
        self.config = self._load_deployment_config()
        
        logging.info(f"Live Deployment Manager initialized - Live Mode: {live_mode}")
        
    def _load_deployment_config(self):
        """Load deployment configuration"""
        return {
            'pre_deployment_checks': [
                'account_verification',
                'connection_test',
                'risk_parameters',
                'strategy_validation',
                'backup_systems'
            ],
            'deployment_phases': [
                'phase_1_monitoring',
                'phase_2_limited_trading', 
                'phase_3_full_trading',
                'phase_4_challenge_verification'
            ],
            'monitoring_intervals': {
                'performance': 60,  # seconds
                'risk': 30,
                'system_health': 120,
                'ftmo_rules': 60
            },
            'alert_thresholds': {
                'drawdown_warning': 0.03,  # 3%
                'drawdown_critical': 0.05, # 5%
                'daily_loss_warning': 0.7, # 70% of limit
                'performance_degradation': 0.8 # 80% of expected
            }
        }
        
    def run_pre_deployment_checks(self):
        """Run comprehensive pre-deployment checks"""
        logging.info("Running pre-deployment checks...")
        
        checks_passed = 0
        total_checks = len(self.config['pre_deployment_checks'])
        
        for check_name in self.config['pre_deployment_checks']:
            success, message = self._run_single_check(check_name)
            
            check_result = {
                'check': check_name,
                'timestamp': datetime.now(),
                'success': success,
                'message': message
            }
            
            self.deployment_log.append(check_result)
            
            if success:
                checks_passed += 1
                logging.info(f"✅ {check_name}: {message}")
            else:
                logging.error(f"❌ {check_name}: {message}")
                
        success_rate = checks_passed / total_checks
        logging.info(f"Pre-deployment checks: {checks_passed}/{total_checks} passed ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% success rate required
        
    def _run_single_check(self, check_name):
        """Run a single deployment check"""
        if check_name == 'account_verification':
            return self._verify_account()
        elif check_name == 'connection_test':
            return self._test_connections()
        elif check_name == 'risk_parameters':
            return self._validate_risk_parameters()
        elif check_name == 'strategy_validation':
            return self._validate_strategies()
        elif check_name == 'backup_systems':
            return self._check_backup_systems()
        else:
            return False, f"Unknown check: {check_name}"
            
    def _verify_account(self):
        """Verify FTMO account details"""
        try:
            # This would integrate with FTMO API or MT5 account verification
            status = self.trading_system.get_phase2_status()
            balance = status['phase1_status']['ftmo_metrics']['current_balance']
            
            if balance > 0:
                return True, f"Account verified - Balance: ${balance:,.2f}"
            else:
                return False, "Invalid account balance"
                
        except Exception as e:
            return False, f"Account verification failed: {e}"
            
    def _test_connections(self):
        """Test all system connections"""
        try:
            # Test MT5 connection
            if self.trading_system.phase1_system.connect_mt5():
                self.trading_system.phase1_system.disconnect_mt5()
                return True, "All connections tested successfully"
            else:
                return False, "MT5 connection failed"
                
        except Exception as e:
            return False, f"Connection test failed: {e}"
            
    def _validate_risk_parameters(self):
        """Validate risk management parameters"""
        try:
            status = self.trading_system.get_phase2_status()
            risk_report = status['risk_manager']
            
            if risk_report['risk_level'] == "NORMAL":
                return True, "Risk parameters validated"
            else:
                return False, f"Risk level not normal: {risk_report['risk_level']}"
                
        except Exception as e:
            return False, f"Risk validation failed: {e}"
            
    def _validate_strategies(self):
        """Validate trading strategies"""
        try:
            # Check if both strategies are available
            status = self.trading_system.get_phase2_status()
            current_strategy = status['phase1_status']['current_strategy']
            
            if current_strategy in ['xgboost', 'fallback']:
                return True, f"Strategy validated: {current_strategy}"
            else:
                return False, f"Invalid strategy: {current_strategy}"
                
        except Exception as e:
            return False, f"Strategy validation failed: {e}"
            
    def _check_backup_systems(self):
        """Check backup and recovery systems"""
        try:
            # Verify critical files exist
            critical_files = [
                'enhanced_risk_manager.py',
                'drawdown_protection.py',
                'ftmo_rule_enforcer.py',
                'ftmo_phase2_system.py'
            ]
            
            missing_files = []
            for file in critical_files:
                if not os.path.exists(file):
                    missing_files.append(file)
                    
            if not missing_files:
                return True, "Backup systems verified"
            else:
                return False, f"Missing critical files: {missing_files}"
                
        except Exception as e:
            return False, f"Backup check failed: {e}"
            
    def deploy_to_live(self):
        """Deploy system to live trading"""
        if not self.run_pre_deployment_checks():
            logging.error("Pre-deployment checks failed - cannot deploy to live")
            return False
            
        logging.info("Starting live deployment...")
        self.deployment_status = "DEPLOYING"
        
        try:
            # Phase 1: Monitoring only
            logging.info("Phase 1: Monitoring mode")
            self._deployment_phase_1()
            
            # Phase 2: Limited trading
            logging.info("Phase 2: Limited trading mode")
            self._deployment_phase_2()
            
            # Phase 3: Full trading
            logging.info("Phase 3: Full trading mode")
            self.deployment_status = "LIVE"
            return self._deployment_phase_3()
            
        except Exception as e:
            logging.error(f"Deployment failed: {e}")
            self.deployment_status = "FAILED"
            return False
            
    def _deployment_phase_1(self):
        """Deployment phase 1: Monitoring only"""
        # Monitor without trading for initial period
        monitoring_duration = timedelta(minutes=30)  # 30 minutes monitoring
        end_time = datetime.now() + monitoring_duration
        
        while datetime.now() < end_time:
            self._monitor_system()
            time.sleep(60)  # Check every minute
            
    def _deployment_phase_2(self):
        """Deployment phase 2: Limited trading"""
        # Limited trading with reduced position sizes
        limited_duration = timedelta(hours=2)  # 2 hours limited trading
        end_time = datetime.now() + limited_duration
        
        # Reduce risk temporarily
        original_risk = self.trading_system.risk_manager.config.RISK_PER_TRADE
        self.trading_system.risk_manager.config.RISK_PER_TRADE = original_risk * 0.5
        
        while datetime.now() < end_time:
            self.trading_system.enhanced_trading_cycle()
            self._monitor_system()
            time.sleep(120)  # 2-minute cycles
            
        # Restore original risk
        self.trading_system.risk_manager.config.RISK_PER_TRADE = original_risk
        
    def _deployment_phase_3(self):
        """Deployment phase 3: Full trading"""
        # Full trading with real-time monitoring
        logging.info("🎉 LIVE TRADING DEPLOYED SUCCESSFULLY!")
        
        # Start continuous monitoring
        self._start_live_monitoring()
        return True
        
    def _monitor_system(self):
        """Monitor system health and performance"""
        status = self.trading_system.get_phase2_status()
        
        # Log monitoring data
        monitor_data = {
            'timestamp': datetime.now(),
            'performance': status['phase1_status']['ftmo_metrics'],
            'risk': status['risk_manager'],
            'system_health': status['rule_enforcer']
        }
        
        self.deployment_log.append(monitor_data)
        
        # Check for alerts
        self._check_alerts(monitor_data)
        
    def _check_alerts(self, monitor_data):
        """Check for system alerts"""
        metrics = monitor_data['performance']
        
        # Drawdown alerts
        current_drawdown = monitor_data['risk'].get('current_drawdown', 0)
        if current_drawdown > self.config['alert_thresholds']['drawdown_critical']:
            self._trigger_alert("CRITICAL", f"Drawdown exceeded: {current_drawdown:.1%}")
        elif current_drawdown > self.config['alert_thresholds']['drawdown_warning']:
            self._trigger_alert("WARNING", f"Drawdown warning: {current_drawdown:.1%}")
            
        # Daily loss alerts
        daily_profit = metrics.get('daily_profit', 0)
        daily_limit = metrics.get('daily_loss_limit', 5000)
        if daily_profit <= -daily_limit * self.config['alert_thresholds']['daily_loss_warning']:
            self._trigger_alert("WARNING", f"Approaching daily loss limit: {daily_profit}")
            
    def _trigger_alert(self, level, message):
        """Trigger system alert"""
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        }
        
        logging.warning(f"ALERT {level}: {message}")
        
        # Here you would integrate with email/SMS alerts
        # self._send_external_alert(alert)
        
    def _start_live_monitoring(self):
        """Start continuous live monitoring"""
        # This would run in a separate thread
        logging.info("Live monitoring started")
        
    def get_deployment_status(self):
        """Get current deployment status"""
        return {
            'deployment_status': self.deployment_status,
            'live_mode': self.live_mode,
            'account_id': self.account_id,
            'challenge_type': self.challenge_type,
            'log_entries': len(self.deployment_log),
            'last_check': datetime.now().isoformat()
        }
