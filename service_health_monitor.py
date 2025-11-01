# Service Health Monitor for Trading System
import psutil
import time
import logging
import zmq
import json
from threading import Thread
from datetime import datetime

class ServiceHealthMonitor:
    def __init__(self, services_config):
        self.services_config = services_config
        self.health_status = {}
        self.failure_count = {service: 0 for service in services_config.keys()}
        self.alert_threshold = 3  # Consecutive failures before alert
        self.recovery_attempts = {}
        
        # ZMQ for health reporting
        self.context = zmq.Context()
        self.health_pub = self.context.socket(zmq.PUB)
        self.health_pub.bind("tcp://*:5559")
        
        # Service status subscribers
        self.status_sub = self.context.socket(zmq.SUB)
        self.status_sub.bind("tcp://*:5560")
        self.status_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        logging.info("Service Health Monitor initialized")
        
    def start_monitoring(self):
        """Start health monitoring in background thread"""
        monitor_thread = Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        return monitor_thread
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Check each service
                for service_name, config in self.services_config.items():
                    health = self._check_service_health(service_name, config)
                    self.health_status[service_name] = health
                    
                    # Trigger recovery if service is critical
                    if health['status'] == 'critical':
                        self._trigger_recovery(service_name, config)
                    elif health['status'] == 'healthy':
                        self.failure_count[service_name] = 0  # Reset failure count
                        
                # Publish health status
                self._publish_health_status()
                
                # Check for service status messages
                self._check_status_messages()
                
                time.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                time.sleep(30)
                
    def _check_service_health(self, service_name, config):
        """Check health of specific service"""
        health = {
            'status': 'unknown', 
            'details': '',
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': 0,
            'memory_usage': 0,
            'response_time': 0
        }
        
        try:
            # Find process by name or PID
            process = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if (config.get('process_name') and config['process_name'] in proc.info['name']) or \
                   (config.get('pid') and config['pid'] == proc.info['pid']):
                    process = proc
                    break
                    
            if process:
                # Get resource usage
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                health['cpu_usage'] = cpu_percent
                health['memory_usage'] = memory_mb
                health['status'] = 'healthy'
                
                # Check resource thresholds
                if cpu_percent > 85:
                    health['status'] = 'warning'
                    health['details'] = f'High CPU: {cpu_percent}%'
                elif memory_mb > config.get('memory_limit', 500):  # 500MB default
                    health['status'] = 'warning' 
                    health['details'] = f'High memory: {memory_mb:.1f}MB'
                    
                # Check service-specific health via ZMQ
                service_health = self._check_service_response(service_name, config)
                if not service_health:
                    health['status'] = 'critical'
                    health['details'] = 'No response from service'
                    
            else:
                health['status'] = 'critical'
                health['details'] = 'Process not found'
                self.failure_count[service_name] += 1
                
        except Exception as e:
            health['status'] = 'critical'
            health['details'] = f'Health check error: {str(e)}'
            self.failure_count[service_name] += 1
            
        # Check if we've exceeded failure threshold
        if self.failure_count[service_name] >= self.alert_threshold:
            health['status'] = 'critical'
            health['details'] += f' | {self.failure_count[service_name]} consecutive failures'
            
        return health
        
    def _check_service_response(self, service_name, config):
        """Check if service is responding via ZMQ"""
        try:
            # Try to ping service via configured port
            req_socket = self.context.socket(zmq.REQ)
            req_socket.setsockopt(zmq.LINGER, 0)
            req_socket.connect(f"tcp://localhost:{config.get('health_port', 5570)}")
            req_socket.send_string("ping")
            
            # Wait for response with timeout
            if req_socket.poll(2000):  # 2 second timeout
                response = req_socket.recv_string()
                return response == "pong"
            return False
        except:
            return False
            
    def _trigger_recovery(self, service_name, config):
        """Attempt to recover a failed service"""
        if service_name not in self.recovery_attempts:
            self.recovery_attempts[service_name] = 0
            
        if self.recovery_attempts[service_name] < 3:  # Max 3 recovery attempts
            logging.warning(f"Attempting recovery for {service_name} (attempt {self.recovery_attempts[service_name] + 1})")
            
            # Implement service-specific recovery logic
            recovery_success = self._execute_recovery_script(service_name, config)
            
            if recovery_success:
                logging.info(f"Service {service_name} recovered successfully")
                self.recovery_attempts[service_name] = 0
            else:
                self.recovery_attempts[service_name] += 1
                logging.error(f"Recovery failed for {service_name}")
        else:
            logging.critical(f"Service {service_name} failed after 3 recovery attempts")
            
    def _execute_recovery_script(self, service_name, config):
        """Execute service recovery script"""
        # This would call external recovery scripts
        # For now, return True for simulation
        return True
        
    def _check_status_messages(self):
        """Check for status messages from services"""
        try:
            message = self.status_sub.recv_string(zmq.NOBLOCK)
            data = json.loads(message)
            service_name = data.get('service_name')
            if service_name:
                self.health_status[service_name] = data
        except zmq.Again:
            pass  # No new messages
            
    def _publish_health_status(self):
        """Publish health status to all subscribers"""
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'services': self.health_status,
            'overall_status': self._calculate_overall_status()
        }
        self.health_pub.send_string(json.dumps(status_report))
        
    def _calculate_overall_status(self):
        """Calculate overall system health status"""
        statuses = [health['status'] for health in self.health_status.values()]
        
        if any(status == 'critical' for status in statuses):
            return 'critical'
        elif any(status == 'warning' for status in statuses):
            return 'warning'
        else:
            return 'healthy'
            
    def get_health_report(self):
        """Return complete health report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'services': self.health_status,
            'overall_status': self._calculate_overall_status(),
            'failure_counts': self.failure_count
        }
