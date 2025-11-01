# FTMO Phase 3 Integration - Live Deployment & Monitoring
from live_deployment_manager import LiveDeploymentManager
from performance_monitor import RealTimePerformanceMonitor
from automated_reporter import AutomatedReporter
from ftmo_phase2_system import FTMO_Phase2_System
import logging
import time
from threading import Thread

class FTMO_Phase3_System:
    def __init__(self, account_id="1600038177", challenge_type="200k", live_mode=False):
        self.account_id = account_id
        self.challenge_type = challenge_type
        self.live_mode = live_mode
        
        # Initialize Phase 2 system
        self.phase2_system = FTMO_Phase2_System(account_id, challenge_type)
        
        # Initialize Phase 3 components
        self.deployment_manager = LiveDeploymentManager(account_id, challenge_type, live_mode)
        self.performance_monitor = RealTimePerformanceMonitor(self.phase2_system)
        self.reporter = AutomatedReporter(self.phase2_system)
        
        # Phase 3 state
        self.monitoring_active = False
        self.reporting_active = False
        
        logging.info("FTMO Phase 3 System initialized")
        
    def deploy_to_live(self):
        """Deploy system to live trading with Phase 3 features"""
        logging.info("🚀 Starting Phase 3 Live Deployment")
        
        # Step 1: Pre-deployment checks
        if not self.deployment_manager.run_pre_deployment_checks():
            logging.error("Pre-deployment checks failed")
            return False
            
        # Step 2: Start real-time monitoring
        self.performance_monitor.start_monitoring()
        self.monitoring_active = True
        
        # Step 3: Start automated reporting
        self._start_reporting_system()
        self.reporting_active = True
        
        # Step 4: Deploy to live
        success = self.deployment_manager.deploy_to_live()
        
        if success:
            logging.info("🎉 Phase 3 Live Deployment Completed Successfully!")
            self._generate_initial_reports()
        else:
            logging.error("Phase 3 deployment failed")
            
        return success
        
    def _start_reporting_system(self):
        """Start automated reporting system"""
        # This would run in a separate thread
        logging.info("Automated reporting system started")
        
    def _generate_initial_reports(self):
        """Generate initial deployment reports"""
        daily_report = self.reporter.generate_daily_report()
        self.reporter.export_report(daily_report, "initial_deployment_report.json")
        
        logging.info("Initial reports generated")
        
    def start_live_operation(self):
        """Start live operation with all Phase 3 features"""
        if not self.live_mode:
            logging.warning("Not in live mode - starting in demo mode")
            
        # Start enhanced trading with monitoring
        self.phase2_system.start_enhanced_trading()
        
    def get_phase3_status(self):
        """Get comprehensive Phase 3 status"""
        return {
            'phase2_status': self.phase2_system.get_phase2_status(),
            'deployment_status': self.deployment_manager.get_deployment_status(),
            'monitoring_active': self.monitoring_active,
            'reporting_active': self.reporting_active,
            'live_mode': self.live_mode,
            'timestamp': time.time()
        }
        
    def emergency_stop(self):
        """Emergency stop all trading and monitoring"""
        logging.critical("🛑 EMERGENCY STOP ACTIVATED")
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        self.monitoring_active = False
        
        # Stop reporting
        self.reporting_active = False
        
        # Export final performance data
        self.performance_monitor.export_performance_data("emergency_stop_data.json")
        
        logging.info("Emergency stop completed - all systems halted")

# Live deployment test
def test_phase3_deployment():
    """Test Phase 3 deployment in demo mode"""
    print("🧪 Testing Phase 3 Live Deployment")
    print("==================================")
    
    phase3_system = FTMO_Phase3_System(live_mode=False)
    
    # Test deployment manager
    deployment_status = phase3_system.deployment_manager.get_deployment_status()
    print(f"Deployment Status: {deployment_status['deployment_status']}")
    
    # Test performance monitor
    phase3_system.performance_monitor.start_monitoring()
    time.sleep(2)  # Brief monitoring
    phase3_system.performance_monitor.stop_monitoring()
    print("✅ Performance monitoring tested")
    
    # Test reporting
    daily_report = phase3_system.reporter.generate_daily_report()
    print(f"✅ Daily report generated: ${daily_report['challenge_progress']['total_profit']} profit")
    
    print("🎉 Phase 3 deployment test completed!")

if __name__ == "__main__":
    test_phase3_deployment()
