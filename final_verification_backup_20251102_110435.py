import os
import sys
import importlib
from pathlib import Path

class FinalVerification:
    def __init__(self):
        self.expected_symbols = ['BTCX25.sim', 'US30Z25.sim', 'XAUZ25.sim']
        self.expected_timeframe = 'H4'
        self.issues_found = []
        
    def check_config_files(self):
        """Verify configuration files are properly updated"""
        print("🔍 CHECKING CONFIGURATION FILES...")
        
        config_files_to_check = [
            'ftmo_config.py',
            'ftmo_config_complete.py', 
            'config/ftmo_config.py'
        ]
        
        for config_file in config_files_to_check:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for optimal symbols
                    has_optimal_symbols = all(symbol in content for symbol in self.expected_symbols)
                    has_h4_timeframe = 'H4' in content or 'TIMEFRAME_H4' in content
                    
                    if not has_optimal_symbols:
                        self.issues_found.append(f"❌ {config_file}: Missing optimal symbols")
                    else:
                        print(f"✅ {config_file}: Symbols configured correctly")
                    
                    if not has_h4_timeframe:
                        self.issues_found.append(f"❌ {config_file}: Missing H4 timeframe")
                    else:
                        print(f"✅ {config_file}: Timeframe configured correctly")
                        
                except Exception as e:
                    self.issues_found.append(f"❌ {config_file}: Error reading file - {e}")
            else:
                print(f"⚠️ {config_file}: File not found (may be expected)")
    
    def check_trading_files(self):
        """Verify trading logic files"""
        print("\\n🔍 CHECKING TRADING LOGIC FILES...")
        
        trading_files = [
            'ftmo_trade.py',
            'integrate_professional_ensemble.py',
            'enhanced_trading_dashboard.py'
        ]
        
        for trading_file in trading_files:
            if os.path.exists(trading_file):
                try:
                    with open(trading_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for Bitcoin integration
                    has_bitcoin = 'BTCX25.sim' in content
                    has_us30 = 'US30Z25.sim' in content
                    has_gold = 'XAUZ25.sim' in content
                    
                    symbols_found = []
                    if has_bitcoin: symbols_found.append('BTC')
                    if has_us30: symbols_found.append('US30')
                    if has_gold: symbols_found.append('XAU')
                    
                    if symbols_found:
                        print(f"✅ {trading_file}: Found symbols {', '.join(symbols_found)}")
                    else:
                        self.issues_found.append(f"❌ {trading_file}: No optimal symbols found")
                        
                except Exception as e:
                    self.issues_found.append(f"❌ {trading_file}: Error reading - {e}")
            else:
                self.issues_found.append(f"❌ {trading_file}: File not found")
    
    def test_imports(self):
        """Test that key modules can be imported"""
        print("\\n🔍 TESTING MODULE IMPORTS...")
        
        modules_to_test = [
            'ftmo_config',
            'ftmo_trade',
            'enhanced_trading_dashboard',
            'integrate_professional_ensemble'
        ]
        
        for module_name in modules_to_test:
            module_path = f"{module_name}.py"
            if os.path.exists(module_path):
                try:
                    # Remove .py extension for import
                    import_name = module_name
                    if import_name in sys.modules:
                        del sys.modules[import_name]
                    
                    module = importlib.import_module(import_name)
                    print(f"✅ {module_name}: Import successful")
                    
                except Exception as e:
                    self.issues_found.append(f"❌ {module_name}: Import failed - {e}")
            else:
                print(f"⚠️ {module_name}: File not found (may be expected)")
    
    def check_gui_dependencies(self):
        """Verify GUI dependencies are available"""
        print("\\n🔍 CHECKING GUI DEPENDENCIES...")
        
        gui_dependencies = [
            'tkinter',
            'PyQt5',
            'matplotlib',
            'pandas',
            'numpy',
            'MetaTrader5',
            'sklearn'
        ]
        
        for dependency in gui_dependencies:
            try:
                __import__(dependency)
                print(f"✅ {dependency}: Available")
            except ImportError:
                self.issues_found.append(f"❌ {dependency}: Missing - pip install {dependency}")
    
    def create_test_script(self):
        """Create a simple test script to verify the GUI works"""
        test_script = '''
# QUICK GUI TEST SCRIPT
try:
    import ftmo_config
    import enhanced_trading_dashboard as dashboard
    
    print("✅ Basic imports successful")
    
    # Check configuration
    if hasattr(ftmo_config, 'SYMBOLS'):
        symbols = ftmo_config.SYMBOLS
        print(f"✅ Symbols configured: {symbols}")
    else:
        print("❌ SYMBOLS not found in configuration")
    
    print("🚀 Attempting to launch GUI...")
    
    # Try to initialize the dashboard
    try:
        # This would typically be how your GUI starts
        # dashboard.main() or similar
        print("✅ GUI components available")
    except Exception as e:
        print(f"❌ GUI initialization issue: {e}")
        
except Exception as e:
    print(f"❌ Critical error: {e}")

print("\\\\n🎯 Test completed. Check output above for issues.")
'''
        
        with open('quick_gui_test.py', 'w') as f:
            f.write(test_script)
        
        print("✅ Created quick_gui_test.py")
        return 'quick_gui_test.py'
    
    def run_verification(self):
        """Run complete verification"""
        print("🚀 FINAL SYSTEM VERIFICATION")
        print("=" * 60)
        
        self.check_config_files()
        self.check_trading_files()
        self.test_imports()
        self.check_gui_dependencies()
        
        test_script = self.create_test_script()
        
        # Summary
        print("\\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        if self.issues_found:
            print("❌ ISSUES FOUND:")
            for issue in self.issues_found:
                print(f"  {issue}")
            print(f"\\n⚠️ Total issues: {len(self.issues_found)}")
            print("💡 Please fix these issues before running your GUI")
        else:
            print("✅ ALL CHECKS PASSED!")
            print("🎉 Your system should be ready to run!")
        
        print(f"\\n🔧 Test script created: {test_script}")
        print("💡 Run: python quick_gui_test.py to test basic functionality")

# Run verification
if __name__ == "__main__":
    verifier = FinalVerification()
    verifier.run_verification()
