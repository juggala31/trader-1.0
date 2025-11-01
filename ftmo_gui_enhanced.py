# FTMO GUI-Enhanced Trading System
import threading
import time
from datetime import datetime
from ftmo_minimal import FTMOMinimalSystem

class FTMO_GUI_Enhanced_System:
    def __init__(self, account_id="1600038177", challenge_type="200k", live_mode=False):
        self.trading_system = FTMOMinimalSystem(account_id, challenge_type, live_mode)
        self.gui_callbacks = []
        self.trading_active = False
        self.trading_thread = None
        
    def register_gui_callback(self, callback):
        """Register GUI callback for real-time updates"""
        self.gui_callbacks.append(callback)
        
    def notify_gui(self, event_type, data):
        """Notify GUI of important events"""
        for callback in self.gui_callbacks:
            try:
                callback(event_type, data)
            except:
                pass  # GUI might not be ready
                
    def start_trading_with_gui(self):
        """Start trading with GUI integration"""
        self.trading_active = True
        
        # Notify GUI that trading is starting
        self.notify_gui('TRADING_START', {'time': datetime.now()})
        
        # Start trading in background thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
    def _trading_loop(self):
        """Main trading loop with GUI updates"""
        while self.trading_active:
            try:
                # Get current system status
                status = self.trading_system.get_system_status()
                
                # Notify GUI of status update
                self.notify_gui('STATUS_UPDATE', status)
                
                # Simulate trading activity (would be real trading logic)
                self._simulate_trading_cycle()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.notify_gui('ERROR', {'message': str(e)})
                time.sleep(10)  # Wait longer on error
                
    def _simulate_trading_cycle(self):
        """Simulate trading cycle for demo purposes"""
        # This would contain actual trading logic
        # For demo, simulate occasional trades
        import random
        
        if random.random() < 0.3:  # 30% chance of trade
            symbols = ['US30', 'US100', 'UAX']
            symbol = random.choice(symbols)
            action = random.choice(['BUY', 'SELL'])
            profit = random.uniform(-100, 200)
            
            trade_data = {
                'symbol': symbol,
                'action': action,
                'profit': profit,
                'time': datetime.now(),
                'status': 'WIN' if profit > 0 else 'LOSS'
            }
            
            self.notify_gui('TRADE_EXECUTED', trade_data)
            
    def stop_trading(self):
        """Stop trading"""
        self.trading_active = False
        self.notify_gui('TRADING_STOP', {'time': datetime.now()})
        
    def get_system_status(self):
        """Get enhanced system status for GUI"""
        base_status = self.trading_system.get_system_status()
        
        # Add GUI-specific status information
        enhanced_status = {
            **base_status,
            'gui_connected': len(self.gui_callbacks) > 0,
            'trading_active': self.trading_active,
            'last_update': datetime.now().isoformat()
        }
        
        return enhanced_status

# Simple test function
def test_gui_system():
    """Test the GUI-enhanced system"""
    print("Testing GUI-Enhanced System...")
    
    system = FTMO_GUI_Enhanced_System()
    
    # Test GUI callback registration
    def test_callback(event, data):
        print(f"GUI Event: {event} - {data}")
        
    system.register_gui_callback(test_callback)
    
    # Test status retrieval
    status = system.get_system_status()
    print(f"System Status: {status['strategy']} strategy, GUI: {status['gui_connected']}")
    
    print("GUI system test completed!")

if __name__ == "__main__":
    test_gui_system()
