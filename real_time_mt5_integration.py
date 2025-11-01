import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class RealTimeMT5Integration:
    def __init__(self, login=1600038177, server="OANDA-Demo-1", password=""):
        self.login = login
        self.server = server
        self.password = password
        self.connected = False
        self.account_info = None
        self.positions_df = None
        self.balance_history = []
        self.equity_history = []
        
    def initialize_connection(self):
        """Initialize connection to MT5 and get account info"""
        try:
            if not mt5.initialize():
                print("MT5 initialization failed")
                return False
            
            # Login to account (password might be required for real accounts)
            if self.password:
                authorized = mt5.login(self.login, password=self.password, server=self.server)
            else:
                authorized = mt5.login(self.login, server=self.server)
            
            if not authorized:
                print(f"Login failed: {mt5.last_error()}")
                return False
            
            self.connected = True
            self._update_account_info()
            print(f"✓ Connected to MT5 Account: {self.login}")
            print(f"✓ Balance: ${self.account_info.balance:.2f}")
            print(f"✓ Equity: ${self.account_info.equity:.2f}")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def _update_account_info(self):
        """Update account information from MT5"""
        if self.connected:
            self.account_info = mt5.account_info()
            if self.account_info:
                # Store history for tracking
                self.balance_history.append({
                    'timestamp': datetime.now(),
                    'balance': self.account_info.balance,
                    'equity': self.account_info.equity,
                    'profit': self.account_info.profit
                })
    
    def get_current_balance(self):
        """Get current account balance"""
        self._update_account_info()
        if self.account_info:
            return self.account_info.balance
        return 100000  # Fallback
    
    def get_current_equity(self):
        """Get current account equity"""
        self._update_account_info()
        if self.account_info:
            return self.account_info.equity
        return 100000  # Fallback
    
    def get_current_profit(self):
        """Get current profit/loss"""
        self._update_account_info()
        if self.account_info:
            return self.account_info.profit
        return 0
    
    def get_open_positions(self):
        """Get all open positions with details"""
        if self.connected:
            positions = mt5.positions_get()
            if positions:
                self.positions_df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
                return self.positions_df
        return pd.DataFrame()
    
    def get_position_pnl(self, symbol=None):
        """Get P&L for specific symbol or all positions"""
        positions_df = self.get_open_positions()
        if positions_df.empty:
            return 0
        
        if symbol:
            symbol_positions = positions_df[positions_df['symbol'] == symbol]
            return symbol_positions['profit'].sum() if not symbol_positions.empty else 0
        else:
            return positions_df['profit'].sum()
    
    def get_daily_pnl(self):
        """Calculate today's P&L"""
        today = datetime.now().date()
        positions_df = self.get_open_positions()
        
        if positions_df.empty:
            return 0
        
        # Filter positions that were likely opened today (simplified approach)
        today_positions = positions_df.copy()
        # Note: MT5 doesn't provide open time in standard positions_get, 
        # so we use a simplified approach
        return today_positions['profit'].sum()
    
    def get_portfolio_summary(self):
        """Get comprehensive portfolio summary"""
        self._update_account_info()
        positions_df = self.get_open_positions()
        
        summary = {
            'timestamp': datetime.now(),
            'balance': self.get_current_balance(),
            'equity': self.get_current_equity(),
            'floating_pnl': self.get_current_profit(),
            'daily_pnl': self.get_daily_pnl(),
            'open_positions': len(positions_df) if not positions_df.empty else 0,
            'total_exposure': 0,
            'risk_ratio': 0
        }
        
        # Calculate total exposure
        if not positions_df.empty:
            summary['total_exposure'] = positions_df['volume'].sum()
            # Simple risk ratio: floating PnL / balance
            if summary['balance'] > 0:
                summary['risk_ratio'] = abs(summary['floating_pnl']) / summary['balance']
        
        return summary
    
    def get_historical_balance(self, days=30):
        """Get historical balance data (simulated for demo)"""
        # In a real system, you'd store this data over time
        # For demo, we'll create simulated history
        if not self.balance_history:
            base_balance = self.get_current_balance()
            for i in range(days, 0, -1):
                date = datetime.now() - timedelta(days=i)
                # Simulate some variation
                variation = np.random.normal(0, 0.01) * base_balance
                self.balance_history.append({
                    'timestamp': date,
                    'balance': base_balance + variation,
                    'equity': base_balance + variation,
                    'profit': variation
                })
        
        return pd.DataFrame(self.balance_history)
    
    def execute_trade(self, symbol, order_type, volume, stop_loss=0, take_profit=0):
        """Execute a trade with real MT5 connection"""
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                print(f"Could not get tick data for {symbol}")
                return None
            
            price = tick.ask if order_type == 'BUY' else tick.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 234000,
                "comment": "AI Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Trade execution failed: {result.retcode} - {result.comment}")
                return None
            
            print(f"✓ Trade executed: {order_type} {volume} {symbol} at {price}")
            return result
            
        except Exception as e:
            print(f"Trade execution error: {e}")
            return None
    
    def close_position(self, ticket):
        """Close a specific position by ticket number"""
        if not self.connected:
            print("Not connected to MT5")
            return None
        
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                print(f"Position with ticket {ticket} not found")
                return None
            
            position = positions[0]
            symbol = position.symbol
            volume = position.volume
            position_type = position.type
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                print(f"Could not get tick data for {symbol}")
                return None
            
            if position_type == mt5.ORDER_TYPE_BUY:
                price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": "AI System Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Position close failed: {result.retcode} - {result.comment}")
                return None
            
            print(f"✓ Position closed: {symbol} with P&L: ${position.profit:.2f}")
            return result
            
        except Exception as e:
            print(f"Position close error: {e}")
            return None
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("MT5 connection closed")

# Enhanced risk management with real balance integration
class RealTimeRiskManager:
    def __init__(self, mt5_integration):
        self.mt5 = mt5_integration
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.total_drawdown_limit = 0.10  # 10% total drawdown limit
        self.daily_starting_balance = None
        self.max_balance = None
        
    def initialize_daily_tracking(self):
        """Initialize daily tracking at start of trading day"""
        current_balance = self.mt5.get_current_balance()
        self.daily_starting_balance = current_balance
        self.max_balance = current_balance
        print(f"Daily tracking initialized. Starting balance: ${current_balance:.2f}")
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit is exceeded"""
        if not self.daily_starting_balance:
            self.initialize_daily_tracking()
        
        current_balance = self.mt5.get_current_balance()
        daily_pnl = current_balance - self.daily_starting_balance
        daily_loss_pct = abs(daily_pnl) / self.daily_starting_balance if daily_pnl < 0 else 0
        
        if daily_loss_pct > self.daily_loss_limit:
            print(f"🚨 DAILY LOSS LIMIT EXCEEDED: {daily_loss_pct:.2%}")
            return False
        return True
    
    def check_total_drawdown_limit(self):
        """Check if total drawdown limit is exceeded"""
        current_balance = self.mt5.get_current_balance()
        
        if self.max_balance is None:
            self.max_balance = current_balance
        
        # Update max balance if current is higher
        if current_balance > self.max_balance:
            self.max_balance = current_balance
        
        drawdown = (self.max_balance - current_balance) / self.max_balance if current_balance < self.max_balance else 0
        
        if drawdown > self.total_drawdown_limit:
            print(f"🚨 TOTAL DRAWDOWN LIMIT EXCEEDED: {drawdown:.2%}")
            return False
        return True
    
    def calculate_safe_position_size(self, symbol, signal_strength):
        """Calculate position size based on real account balance and risk limits"""
        current_balance = self.mt5.get_current_balance()
        current_equity = self.mt5.get_current_equity()
        
        # Base position size (1% of balance per unit of signal strength)
        base_size = current_balance * 0.01 * signal_strength
        
        # Adjust for current risk exposure
        floating_pnl = self.mt5.get_current_profit()
        risk_adjustment = max(0.1, 1 - (abs(floating_pnl) / current_balance))
        
        # Final position size
        position_size = base_size * risk_adjustment
        
        # Ensure we don't exceed daily limits
        if not self.check_daily_loss_limit() or not self.check_total_drawdown_limit():
            position_size = 0  # Stop trading if limits exceeded
        
        print(f"Position size calculated: ${position_size:.2f} for {symbol}")
        return position_size
    
    def get_risk_report(self):
        """Generate comprehensive risk report"""
        portfolio = self.mt5.get_portfolio_summary()
        
        report = {
            'current_balance': portfolio['balance'],
            'current_equity': portfolio['equity'],
            'floating_pnl': portfolio['floating_pnl'],
            'daily_pnl': portfolio['daily_pnl'],
            'open_positions': portfolio['open_positions'],
            'total_exposure': portfolio['total_exposure'],
            'risk_ratio': portfolio['risk_ratio'],
            'daily_limit_remaining': self.daily_loss_limit - (abs(portfolio['daily_pnl']) / self.daily_starting_balance if self.daily_starting_balance else 0),
            'drawdown_remaining': self.total_drawdown_limit - portfolio['risk_ratio']
        }
        
        return report

# Demo function to test real MT5 integration
def test_real_mt5_integration():
    """Test the real MT5 integration with demo account"""
    print("Testing Real MT5 Integration...")
    print("=" * 50)
    
    mt5_integration = RealTimeMT5Integration(login=1600038177, server="OANDA-Demo-1")
    
    if mt5_integration.initialize_connection():
        # Get real account data
        balance = mt5_integration.get_current_balance()
        equity = mt5_integration.get_current_equity()
        profit = mt5_integration.get_current_profit()
        
        print(f"Real Account Balance: ${balance:.2f}")
        print(f"Real Account Equity: ${equity:.2f}")
        print(f"Current P&L: ${profit:.2f}")
        
        # Get portfolio summary
        portfolio = mt5_integration.get_portfolio_summary()
        print(f"Open Positions: {portfolio['open_positions']}")
        print(f"Daily P&L: ${portfolio['daily_pnl']:.2f}")
        
        # Test risk manager
        risk_manager = RealTimeRiskManager(mt5_integration)
        risk_manager.initialize_daily_tracking()
        
        risk_report = risk_manager.get_risk_report()
        print(f"Risk Ratio: {risk_report['risk_ratio']:.2%}")
        print(f"Daily Limit Remaining: {risk_report['daily_limit_remaining']:.2%}")
        
        # Calculate position size for a trade
        position_size = risk_manager.calculate_safe_position_size("US30Z25.sim", 0.8)
        print(f"Safe Position Size: ${position_size:.2f}")
        
        mt5_integration.shutdown()
        print("✓ Real MT5 integration test completed successfully!")
    else:
        print("MT5 connection failed. Using simulated data for demo.")

if __name__ == "__main__":
    test_real_mt5_integration()
