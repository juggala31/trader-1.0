# ZMQ Port Manager - Handle port conflicts for FTMO system
import psutil
import os
import signal
from ftmo_config import FTMOConfig

class PortManager:
    def __init__(self):
        self.config = FTMOConfig()
        
    def cleanup_ports(self):
        """Clean up any processes using FTMO ports"""
        ports = list(self.config.ZMQ_PORTS.values())
        print(f"Checking ports: {ports}")
        
        cleaned = 0
        for port in ports:
            if self.is_port_in_use(port):
                print(f"Port {port} is in use - attempting cleanup...")
                if self.kill_process_using_port(port):
                    cleaned += 1
                    print(f"? Freed port {port}")
                else:
                    print(f"? Could not free port {port}")
                    
        return cleaned
        
    def is_port_in_use(self, port):
        """Check if a port is in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
            
    def kill_process_using_port(self, port):
        """Kill process using a specific port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.info.get('connections', []):
                        if conn.laddr.port == port:
                            print(f"Killing process {proc.info['pid']} using port {port}")
                            os.kill(proc.info['pid'], signal.SIGTERM)
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error killing process: {e}")
            
        return False

def setup_ports():
    """Set up ZMQ ports for FTMO system"""
    port_mgr = PortManager()
    cleaned = port_mgr.cleanup_ports()
    print(f"Cleaned up {cleaned} ports")
    return cleaned

if __name__ == "__main__":
    setup_ports()
