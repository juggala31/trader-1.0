# Advanced Port Manager - Resolves all port conflicts
import psutil
import os
import signal
import socket
import time
from ftmo_config_complete import FTMOConfig

class AdvancedPortManager:
    def __init__(self):
        self.config = FTMOConfig()
        self.port_range = range(5556, 5570)  # All possible FTMO ports
        
    def cleanup_all_ftmo_ports(self):
        """Clean up all FTMO-related ports"""
        print("🧹 Cleaning up FTMO ports...")
        
        ports_freed = 0
        for port in self.port_range:
            if self.is_port_in_use(port):
                print(f"Port {port} is in use - attempting cleanup...")
                if self.kill_process_using_port(port):
                    ports_freed += 1
                    print(f"✓ Freed port {port}")
                else:
                    print(f"✗ Could not free port {port}")
                    
        print(f"Freed {ports_freed} ports")
        return ports_freed
        
    def is_port_in_use(self, port):
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
            
    def kill_process_using_port(self, port):
        """Kill process using a specific port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.info.get('connections', []):
                        if hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                            print(f"Killing process {proc.info['pid']} using port {port}")
                            os.kill(proc.info['pid'], signal.SIGTERM)
                            time.sleep(1)  # Give it time to die
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                    continue
        except Exception as e:
            print(f"Error killing process: {e}")
            
        return False
        
    def get_available_port(self, base_port):
        """Get an available port starting from base_port"""
        for port in range(base_port, base_port + 10):
            if not self.is_port_in_use(port):
                return port
        return base_port  # Fallback
        
    def setup_ports_for_system(self, system_name):
        """Set up unique ports for each system instance"""
        port_mapping = {}
        
        for port_name, base_port in self.config.ZMQ_PORTS.items():
            available_port = self.get_available_port(base_port)
            port_mapping[port_name] = available_port
            
        print(f"Port mapping for {system_name}: {port_mapping}")
        return port_mapping

def setup_ftmo_environment():
    """Set up clean FTMO environment"""
    port_mgr = AdvancedPortManager()
    
    # Clean up any existing ports
    ports_freed = port_mgr.cleanup_all_ftmo_ports()
    
    # Wait for cleanup to complete
    time.sleep(2)
    
    print("✅ FTMO environment ready")
    return port_mgr

if __name__ == "__main__":
    setup_ftmo_environment()
