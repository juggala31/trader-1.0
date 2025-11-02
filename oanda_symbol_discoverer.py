import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

class OandaSymbolDiscoverer:
    def __init__(self):
        self.timeframes = {
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
    def discover_oanda_symbols(self):
        """Discover what symbols are actually available in OANDA MT5"""
        try:
            if not mt5.initialize():
                print("❌ MT5 initialization failed")
                return False
            
            # Get ALL available symbols
            all_symbols = mt5.symbols_get()
            print(f"✅ MT5 connected - {len(all_symbols)} total symbols available")
            
            # Show ALL symbols for debugging
            print("\n📋 ALL AVAILABLE SYMBOLS:")
            print("-" * 50)
            for i, symbol in enumerate(all_symbols[:50]):  # Show first 50
                print(f"{i+1:2d}. {symbol.name}")
            
            if len(all_symbols) > 50:
                print(f"... and {len(all_symbols) - 50} more symbols")
            
            # Categorize symbols automatically
            precious_metals = []
            indices = []
            forex_pairs = []
            commodities = []
            cryptocurrencies = []
            other = []
            
            for symbol in all_symbols:
                name = symbol.name.upper()
                
                # Precious metals detection
                if any(keyword in name for keyword in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM']):
                    precious_metals.append(symbol.name)
                
                # Indices detection
                elif any(keyword in name for keyword in ['US30', 'US500', 'US100', 'SPX', 'NAS', 'DOW', 'DAX', 'FTSE', 'NIKKEI', 'CAC', 'AUS200', 'HSI', 'SENSEX']):
                    indices.append(symbol.name)
                
                # Forex pairs detection
                elif any(keyword in name for keyword in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']) and len(name) == 6:
                    forex_pairs.append(symbol.name)
                
                # Commodities detection
                elif any(keyword in name for keyword in ['OIL', 'BRENT', 'WTI', 'COPPER', 'NATURAL', 'GAS']):
                    commodities.append(symbol.name)
                
                # Cryptocurrencies detection
                elif any(keyword in name for keyword in ['BTC', 'ETH', 'XRP', 'LTC', 'BCH']):
                    cryptocurrencies.append(symbol.name)
                
                else:
                    other.append(symbol.name)
            
            # Display categorized symbols
            print(f"\n💎 PRECIOUS METALS ({len(precious_metals)}):")
            for pm in precious_metals:
                print(f"   ✅ {pm}")
            
            print(f"\n📈 INDICES ({len(indices)}):")
            for idx in indices:
                print(f"   ✅ {idx}")
            
            print(f"\n💱 FOREX PAIRS ({len(forex_pairs)}):")
            for fx in forex_pairs[:10]:  # Show first 10
                print(f"   ✅ {fx}")
            if len(forex_pairs) > 10:
                print(f"   ... and {len(forex_pairs) - 10} more")
            
            print(f"\n🛢️ COMMODITIES ({len(commodities)}):")
            for cmd in commodities:
                print(f"   ✅ {cmd}")
            
            print(f"\n₿ CRYPTOCURRENCIES ({len(cryptocurrencies)}):")
            for crypto in cryptocurrencies:
                print(f"   ✅ {crypto}")
            
            print(f"\n❓ OTHER SYMBOLS ({len(other)}):")
            for sym in other[:10]:
                print(f"   ❓ {sym}")
            
            # Test data availability for top symbols
            print(f"\n🔍 TESTING DATA AVAILABILITY:")
            print("-" * 50)
            
            # Test symbols from each category
            test_symbols = precious_metals[:3] + indices[:3] + forex_pairs[:2] + commodities[:2]
            
            successful_symbols = []
            
            for symbol in test_symbols:
                print(f"Testing {symbol}...")
                df = self.get_test_data(symbol, mt5.TIMEFRAME_H1)
                if df is not None and len(df) > 100:
                    successful_symbols.append(symbol)
                    print(f"   ✅ {symbol}: {len(df)} bars available")
                else:
                    print(f"   ❌ {symbol}: No sufficient data")
            
            print(f"\n🎯 RECOMMENDED SYMBOLS FOR TESTING:")
            for symbol in successful_symbols:
                print(f"   📊 {symbol}")
            
            # Save symbol list for future use
            self.save_symbol_list(precious_metals, indices, forex_pairs, commodities, cryptocurrencies)
            
            return True
            
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    def get_test_data(self, symbol, timeframe, bars=500):
        """Test if symbol has sufficient data"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except:
            return None
    
    def save_symbol_list(self, precious_metals, indices, forex_pairs, commodities, cryptocurrencies):
        """Save discovered symbols to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"oanda_symbols_discovered_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("OANDA MT5 SYMBOL DISCOVERY REPORT\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"Discovery Date: {datetime.now()}\\n\\n")
            
            f.write("PRECIOUS METALS:\\n")
            f.write("-" * 30 + "\\n")
            for symbol in precious_metals:
                f.write(f"{symbol}\\n")
            
            f.write("\\nINDICES:\\n")
            f.write("-" * 30 + "\\n")
            for symbol in indices:
                f.write(f"{symbol}\\n")
            
            f.write("\\nFOREX PAIRS:\\n")
            f.write("-" * 30 + "\\n")
            for symbol in forex_pairs:
                f.write(f"{symbol}\\n")
            
            f.write("\\nCOMMODITIES:\\n")
            f.write("-" * 30 + "\\n")
            for symbol in commodities:
                f.write(f"{symbol}\\n")
            
            f.write("\\nCRYPTOCURRENCIES:\\n")
            f.write("-" * 30 + "\\n")
            for symbol in cryptocurrencies:
                f.write(f"{symbol}\\n")
        
        print(f"\\n💾 Symbol list saved to: {filename}")
        print("💡 Use these symbols in your trading system!")

# Execute discovery
if __name__ == "__main__":
    print("🔍 OANDA SYMBOL DISCOVERY TOOL")
    print("=" * 60)
    print("This will discover what symbols are actually available in your OANDA account")
    print("=" * 60)
    
    discoverer = OandaSymbolDiscoverer()
    success = discoverer.discover_oanda_symbols()
    
    if success:
        print("\\n✅ Symbol discovery completed!")
        print("Now you know exactly what symbols to use in your AI trading system")
    else:
        print("\\n❌ Discovery failed")
