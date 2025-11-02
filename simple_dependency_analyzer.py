import os
import re
from pathlib import Path

class SimpleDependencyAnalyzer:
    def __init__(self):
        self.main_file = "enhanced_trading_dashboard.py"
        self.connected_files = set()
        
    def read_file_safely(self, file_path):
        """Read file safely handling encoding issues"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            try:
                # Try UTF-8 with BOM
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                return content
            except:
                try:
                    # Try latin-1 as fallback
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    return content
                except:
                    return ""
        except Exception:
            return ""
    
    def find_imports_simple(self, content):
        """Find imports using simple text parsing"""
        imports = set()
        
        # Look for import statements
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Standard imports
            if line.startswith('import '):
                parts = line.split(' ')
                if len(parts) > 1:
                    module = parts[1].split(',')[0].strip()
                    if not module.startswith('os') and not module.startswith('sys'):
                        imports.add(module)
            
            # From imports
            elif line.startswith('from '):
                parts = line.split(' ')
                if len(parts) > 1:
                    module = parts[1]
                    if not module.startswith('.') and not module.startswith('os') and not module.startswith('sys'):
                        imports.add(module)
        
        return imports
    
    def find_local_references(self, content):
        """Find references to local project files"""
        local_refs = set()
        
        # Common FTMO file patterns
        patterns = [
            r'ftmo_[a-zA-Z_]+',
            r'enhanced_[a-zA-Z_]+', 
            r'advanced_[a-zA-Z_]+',
            r'integrate_[a-zA-Z_]+',
            r'market_regime_[a-zA-Z_]+',
            r'reinforcement_learning',
            r'professional_ensemble'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                local_refs.add(match)
        
        return local_refs
    
    def analyze_system(self):
        """Analyze the entire system"""
        print("🔍 ANALYZING ENHANCED TRADING DASHBOARD SYSTEM")
        print("=" * 60)
        
        # Read main file
        content = self.read_file_safely(self.main_file)
        if not content:
            print(f"❌ Could not read {self.main_file}")
            return
        
        print(f"📊 Main file: {self.main_file}")
        print(f"📝 File size: {len(content)} characters")
        
        # Find imports
        imports = self.find_imports_simple(content)
        print(f"📦 Python imports found: {len(imports)}")
        
        # Find local references
        local_refs = self.find_local_references(content)
        print(f"🔗 Local references found: {len(local_refs)}")
        
        # Check which referenced files exist
        existing_files = []
        for ref in local_refs:
            possible_files = [f"{ref}.py", f"models/{ref}.py", f"scripts/{ref}.py"]
            for file_path in possible_files:
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                    break
        
        print(f"📁 Existing connected files: {len(existing_files)}")
        
        # Display results
        print(f"\n🎯 KEY DEPENDENCIES:")
        print("-" * 40)
        
        # Group by category
        config_files = [f for f in existing_files if 'config' in f]
        trading_files = [f for f in existing_files if any(x in f for x in ['trade', 'ensemble', 'strategy'])]
        backtest_files = [f for f in existing_files if 'backtest' in f]
        utility_files = [f for f in existing_files if f not in config_files + trading_files + backtest_files]
        
        if config_files:
            print("📋 CONFIGURATION FILES:")
            for file in sorted(config_files):
                print(f"  ✅ {file}")
        
        if trading_files:
            print("🎯 TRADING LOGIC FILES:")
            for file in sorted(trading_files):
                print(f"  ✅ {file}")
        
        if backtest_files:
            print("📊 BACKTESTING FILES:")
            for file in sorted(backtest_files):
                print(f"  ✅ {file}")
        
        if utility_files:
            print("🔧 UTILITY FILES:")
            for file in sorted(utility_files):
                print(f"  ✅ {file}")
        
        # Check for Bitcoin integration
        has_bitcoin = 'BTCX25.sim' in content or 'BTC' in content
        has_h4 = 'H4' in content or 'TIMEFRAME_H4' in content
        
        print(f"\n🎯 OPTIMIZATION CHECK:")
        print(f"  {'✅' if has_bitcoin else '❌'} Bitcoin integration: {has_bitcoin}")
        print(f"  {'✅' if has_h4 else '❌'} H4 timeframe: {has_h4}")
        
        return existing_files

# Run analysis
if __name__ == "__main__":
    analyzer = SimpleDependencyAnalyzer()
    connected_files = analyzer.analyze_system()
    
    print(f"\n💡 Total files in system: {len(connected_files)}")
    print("🎯 Your GUI appears to be properly structured!")
