# FTMO MetaTrader5 Integration Bridge
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from strategy_orchestrator import StrategyOrchestrator
from ftmo_challenge_logger import FTMOChallengeLogger

class FTMO_MT5_Integration:
    def __init__(self, account_id="1600038177", challenge_type="200k"):
        self.account_id = account_id
        self.challenge_type = challenge_type
        
        # FTMO Components
        class FTMOConfig:
            ZMQ_PORTS = {'strategy_bus': 5556}
            MODEL_PATH = "models/xgboost_model.pkl"
            
        self.config = FTMOConfig()
        self.orchestrator = StrategyOrchestrator(self.config)
        self.ftmo_logger = FTMOChallengeLogger(account_id, challenge_type)
        
        # MT5 Connection
        self.connected = False
        self.symbols = ["US30", "US100", "UAX"]
        self.current_positions = {}
        
        # Trading parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        
    def connect_mt5(self):
        """Connect to MetaTrader5"""
        try:
            if not mt5.initialize():
                print("MT5 initialization failed")
                return False
                
            self.connected = True
            print("✅ Connected to MetaTrader5")
            return True
        except Exception as e:
            print(f"MT5 connection error: {e}")
            return False
            
    def disconnect_mt5(self):
        """Disconnect from MetaTrader5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MetaTrader5")
            
    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            return None
            
        account_info = mt5.account_info()
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'currency': account_info.currency
        }
        
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_M15, count=100):
        """Get market data for a symbol"""
        if not self.connected:
            return None
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, count, 1)
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1]
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for trading decisions"""
        # EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        df['atr'] = self.calculate_atr(df)
        
        return df
        
    def get_trading_signal(self, symbol):
        """Get trading signal based on current strategy"""
        # Get market data
        df = self.get_market_data(symbol)
        if df is None:
            return None
            
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        current_bar = df.iloc[-1]
        
        # Determine signal based on current strategy
        if self.orchestrator.current_strategy == "xgboost":
            signal = self._get_xgboost_signal(df)
        else:
            signal = self._get_fallback_signal(df)
            
        return signal
        
    def _get_xgboost_signal(self, df):
        """XGBoost-based trading signal"""
        # Placeholder - integrate your XGBoost model here
        # For now, use a simple EMA crossover
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if current['ema_20'] > current['ema_50'] and prev['ema_20'] <= prev['ema_50']:
            return {'action': 'BUY', 'confidence': 0.7}
        elif current['ema_20'] < current['ema_50'] and prev['ema_20'] >= prev['ema_50']:
            return {'action': 'SELL', 'confidence': 0.7}
            
        return {'action': 'HOLD', 'confidence': 0.5}
        
    def _get_fallback_signal(self, df):
        """Fallback EMA/RSI strategy signal"""
        current = df.iloc[-1]
        
        # EMA trend
        ema_trend = 1 if current['ema_20'] > current['ema_50'] else -1
        
        # RSI conditions
        if current['rsi'] < 30:
            rsi_signal = 1  # Oversold - potential buy
        elif current['rsi'] > 70:
            rsi_signal = -1  # Overbought - potential sell
        else:
            rsi_signal = 0
            
        # Combined signal
        if ema_trend == 1 and rsi_signal >= 0:
            return {'action': 'BUY', 'confidence': 0.6}
        elif ema_trend == -1 and rsi_signal <= 0:
            return {'action': 'SELL', 'confidence': 0.6}
            
        return {'action': 'HOLD', 'confidence': 0.4}
        
    def execute_trade(self, symbol, signal):
        """Execute a trade on MetaTrader5"""
        if signal['action'] == 'HOLD':
            return None
            
        # Calculate position size based on risk
        account_info = self.get_account_info()
        if account_info is None:
            return None
            
        # Get current price and ATR for stop loss
        df = self.get_market_data(symbol)
        atr = self.calculate_atr(df)
        
        # Calculate position size (simplified)
        risk_amount = account_info['balance'] * self.risk_per_trade
        point_value = 1.0  # Adjust based on symbol
        stop_loss_points = atr * self.sl_atr_multiplier
        volume = risk_amount / stop_loss_points
        
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal['action'] == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).ask if signal['action'] == 'BUY' else mt5.symbol_info_tick(symbol).bid,
            "sl": 0,  # Calculate based on ATR
            "tp": 0,  # Calculate based on ATR
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send trade request
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Log successful trade
            trade_data = {
                'symbol': symbol,
                'type': signal['action'],
                'volume': volume,
                'profit': 0,  # Will be updated later
                'strategy': self.orchestrator.current_strategy,
                'confidence': signal['confidence']
            }
            
            self.ftmo_logger.log_trade(trade_data)
            print(f"✅ Trade executed: {signal['action']} {symbol}")
            
        return result
        
    def run_trading_cycle(self):
        """Run one trading cycle"""
        if not self.connected:
            print("Not connected to MT5")
            return
            
        for symbol in self.symbols:
            signal = self.get_trading_signal(symbol)
            if signal and signal['action'] != 'HOLD':
                self.execute_trade(symbol, signal)
                
    def start_trading(self):
        """Start the automated trading system"""
        print("🚀 Starting FTMO MT5 Trading System")
        print("====================================")
        
        if not self.connect_mt5():
            return False
            
        try:
            while True:
                self.run_trading_cycle()
                
                # Check FTMO rules
                if self.ftmo_logger.metrics['daily_profit'] < -self.ftmo_logger.metrics['daily_loss_limit']:
                    print("⚠️ Daily loss limit reached - pausing trading")
                    time.sleep(300)  # Wait 5 minutes
                    
                time.sleep(60)  # Wait 1 minute between cycles
                
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        finally:
            self.disconnect_mt5()
            
    def get_system_status(self):
        """Get system status"""
        return {
            'connected': self.connected,
            'current_strategy': self.orchestrator.current_strategy,
            'ftmo_metrics': self.ftmo_logger.metrics,
            'account_info': self.get_account_info()
        }

# Quick test function
def test_integration():
    """Test the MT5 integration"""
    print("Testing MT5 Integration...")
    
    ftmo_mt5 = FTMO_MT5_Integration()
    
    # Test connection
    if ftmo_mt5.connect_mt5():
        print("✅ MT5 Connection: OK")
        
        # Test account info
        account_info = ftmo_mt5.get_account_info()
        if account_info:
            print(f"✅ Account Info: {account_info['balance']} {account_info['currency']}")
            
        # Test market data
        for symbol in ftmo_mt5.symbols:
            data = ftmo_mt5.get_market_data(symbol, count=10)
            if data is not None:
                print(f"✅ {symbol} Data: {len(data)} bars")
                
        ftmo_mt5.disconnect_mt5()
        
    # Test FTMO components
    status = ftmo_mt5.get_system_status()
    print(f"✅ FTMO Status: {status['current_strategy']} strategy")
    
    print("🎉 Integration test completed!")

if __name__ == "__main__":
    test_integration()
