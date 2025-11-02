# enhanced_mt5_integration.py - FIXED for OANDA compatibility
import MetaTrader5 as mt5
import time
import logging
from datetime import datetime
import threading
from queue import Queue

logger = logging.getLogger('FTMO_AI')

class ProfessionalMT5Manager:
    """Professional MT5 integration with OANDA compatibility"""
    
    def __init__(self, login: int, password: str, server: str, terminal_path: str):
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        self.is_connected = False
        self.connection_attempts = 0
        self.max_attempts = 5
        self.reconnect_delay = 30
        
        # Order execution monitoring
        self.order_queue = Queue()
        self.execution_monitor = None
        
        # Performance tracking
        self.connection_stats = {
            'total_connections': 0,
            'failed_connections': 0,
            'last_connection': None
        }
    
    def initialize_connection(self):
        """Professional MT5 connection with robust error handling"""
        logger.info("🔌 Initializing professional MT5 connection...")
        
        try:
            # Close any existing connection
            if mt5.initialize():
                mt5.shutdown()
                time.sleep(2)
            
            # Attempt connection with retry logic
            for attempt in range(self.max_attempts):
                logger.info(f"Connection attempt {attempt + 1}/{self.max_attempts}")
                
                try:
                    connected = mt5.initialize(
                        path=self.terminal_path,
                        login=self.login,
                        password=self.password,
                        server=self.server,
                        timeout=30000
                    )
                    
                    if connected:
                        self.is_connected = True
                        self.connection_attempts = 0
                        self.connection_stats['total_connections'] += 1
                        self.connection_stats['last_connection'] = datetime.now()
                        
                        # Verify account info
                        account_info = mt5.account_info()
                        if account_info:
                            logger.info(f"✅ MT5 connected successfully - Account: {account_info.login}")
                            return True
                        else:
                            logger.warning("⚠️ Connected but could not retrieve account info")
                            mt5.shutdown()
                            continue
                    
                except Exception as e:
                    logger.error(f"❌ Connection attempt {attempt + 1} failed: {e}")
                    self.connection_stats['failed_connections'] += 1
                
                # Wait before retry
                if attempt < self.max_attempts - 1:
                    time.sleep(self.reconnect_delay)
            
            logger.error("❌ All connection attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection initialization error: {e}")
            return False
    
    def execute_trade_professional(self, symbol, order_type, volume, sl, tp, comment=""):
        """Professional trade execution with enhanced error handling"""
        if not self.is_connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        try:
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": self._get_order_type(order_type),
                "price": self._get_current_price(symbol, order_type),
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"FTMO AI - {comment}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order with retry logic
            result = self._send_order_with_retry(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Trade executed: {symbol} {order_type} {volume} lots")
                return {
                    'success': True,
                    'ticket': result.order,
                    'price': result.price,
                    'volume': result.volume
                }
            else:
                error_msg = self._get_error_message(result.retcode) if result else "Unknown error"
                logger.error(f"❌ Trade execution failed: {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_order_with_retry(self, request, max_retries=3):
        """Send order with retry logic for common issues"""
        for attempt in range(max_retries):
            try:
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    return result
                
                # Retry for certain error codes
                if result and result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_TIMEOUT]:
                    logger.warning(f"⚠️ Retry {attempt + 1} for error: {self._get_error_message(result.retcode)}")
                    time.sleep(1)
                    continue
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Order send attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
        
        return None
    
    def close_position_professional(self, ticket, symbol, volume):
        """Professional position closing"""
        if not self.is_connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {'success': False, 'error': 'Position not found'}
            
            position = positions[0]
            
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "FTMO AI - Close Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position closed: {symbol} Ticket: {ticket}")
                return {
                    'success': True,
                    'ticket': result.order,
                    'price': result.price
                }
            else:
                error_msg = self._get_error_message(result.retcode) if result else "Unknown error"
                logger.error(f"❌ Position close failed: {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"❌ Position close error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_market_data_professional(self, symbol, timeframe, count):
        """Professional market data retrieval"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {'success': False, 'error': 'No tick data available'}
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None:
                return {'success': False, 'error': 'No historical data available'}
            
            return {
                'success': True,
                'tick': {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume
                },
                'historical': rates
            }
            
        except Exception as e:
            logger.error(f"❌ Market data error for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_order_type(self, order_type):
        """Convert string order type to MT5 constant"""
        order_map = {
            'buy': mt5.ORDER_TYPE_BUY,
            'sell': mt5.ORDER_TYPE_SELL
        }
        return order_map.get(order_type.lower(), mt5.ORDER_TYPE_BUY)
    
    def _get_current_price(self, symbol, order_type):
        """Get current price based on order type"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return 0.0
        
        if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
            return tick.ask
        else:
            return tick.bid
    
    def _get_error_message(self, retcode):
        """Convert MT5 error code to human-readable message"""
        error_messages = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote - price changed",
            mt5.TRADE_RETCODE_REJECT: "Order rejected",
            mt5.TRADE_RETCODE_DONE: "Order executed successfully",
            mt5.TRADE_RETCODE_ERROR: "Common error",
            mt5.TRADE_RETCODE_TIMEOUT: "Timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_NO_MONEY: "Insufficient funds",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed"
        }
        return error_messages.get(retcode, f"Error code: {retcode}")
    
    def shutdown(self):
        """Professional shutdown with cleanup"""
        logger.info("🔌 Shutting down professional MT5 connection...")
        self.is_connected = False
        
        try:
            if mt5.initialize():
                mt5.shutdown()
                logger.info("✅ MT5 connection closed gracefully")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

class MT5PerformanceMonitor:
    """Monitor MT5 connection and trading performance"""
    
    def __init__(self, mt5_manager):
        self.mt5_manager = mt5_manager
        self.performance_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0
        }
    
    def record_trade(self, success, profit=0.0):
        """Record trade performance"""
        self.performance_stats['total_trades'] += 1
        
        if success:
            self.performance_stats['successful_trades'] += 1
            self.performance_stats['total_profit'] += profit
        else:
            self.performance_stats['failed_trades'] += 1
    
    def get_performance_report(self):
        """Get comprehensive performance report"""
        success_rate = (self.performance_stats['successful_trades'] / self.performance_stats['total_trades'] * 100) if self.performance_stats['total_trades'] > 0 else 0
        
        return {
            'success_rate': success_rate,
            'total_profit': self.performance_stats['total_profit'],
            'trade_count': self.performance_stats['total_trades']
        }
