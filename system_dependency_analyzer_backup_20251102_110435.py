import ast
import os
import sys
from pathlib import Path

class SystemDependencyAnalyzer:
    def __init__(self):
        self.main_file = "enhanced_trading_dashboard.py"
        self.dependencies = set()
        self.import_errors = []
        
    def extract_imports_from_file(self, file_path):
        """Extract all import statements from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.dependencies.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.dependencies.add(node.module)
            
            return True
        except Exception as e:
            self.import_errors.append(f"Error parsing {file_path}: {e}")
            return False
    
    def find_local_imports(self, file_path):
        """Find imports of local project files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for import patterns
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Look for from ... import and import statements
                if line.startswith('from ') or line.startswith('import '):
                    # Check for local file imports (without dots or with relative)
                    if 'ftmo_' in line or 'enhanced_' in line or 'advanced_' in line:
                        # Extract the module name
                        if line.startswith('from '):
                            parts = line.split(' ')
                            if len(parts) > 1:
                                module = parts[1]
                                if not module.startswith('.') and not module.startswith('os') and not module.startswith('sys'):
                                    self.dependencies.add(module)
                        elif line.startswith('import '):
                            parts = line.split(' ')
                            if len(parts) > 1:
                                module = parts[1]
                                if not module.startswith('os') and not module.startswith('sys'):
                                    self.dependencies.add(module)
            
            return True
        except Exception as e:
            self.import_errors.append(f"Error reading {file_path}: {e}")
            return False
    
    def analyze_main_file(self):
        """Analyze the main dashboard file"""
        print(f"🔍 ANALYZING MAIN FILE: {self.main_file}")
        print("=" * 60)
        
        if not os.path.exists(self.main_file):
            print(f"❌ Main file not found: {self.main_file}")
            return False
        
        # Extract standard imports
        self.extract_imports_from_file(self.main_file)
        
        # Find local imports
        self.find_local_imports(self.main_file)
        
        # Also look for dynamic imports (like importlib)
        with open(self.main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for common FTMO module patterns
        ftmo_patterns = [
            'ftmo_', 'enhanced_', 'advanced_', 'integrate_', 
            'market_regime', 'reinforcement_learning', 'professional_ensemble'
        ]
        
        for pattern in ftmo_patterns:
            if pattern in content:
                # Find the actual module names
                lines = content.split('\n')
                for line in lines:
                    if pattern in line and ('import' in line or 'from' in line):
                        print(f"📥 Found import: {line.strip()}")
        
        return True
    
    def find_connected_files(self):
        """Find all files connected to the main dashboard"""
        print(f"\n🔗 FINDING CONNECTED FILES...")
        
        connected_files = set()
        
        # Common FTMO project files that are likely connected
        common_ftmo_files = [
            'ftmo_config.py', 'ftmo_config_complete.py',
            'ftmo_trade.py', 'ftmo_integrated_system.py',
            'integrate_professional_ensemble.py', 'enhanced_mt5_integration.py',
            'advanced_backtester.py', 'market_regime_detector.py',
            'reinforcement_learning.py', 'enhanced_risk_manager.py'
        ]
        
        # Check which of these files exist and might be imported
        for file in common_ftmo_files:
            if os.path.exists(file):
                # Check if this file is referenced in the main file
                with open(self.main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove .py extension for import matching
                module_name = file.replace('.py', '')
                if module_name in content:
                    connected_files.add(file)
                    print(f"✅ Connected: {file}")
        
        return connected_files
    
    def analyze_import_chain(self, file_path, depth=0, max_depth=3):
        """Recursively analyze import chains"""
        if depth > max_depth:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract imports from this file
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        # Check if this is a local module
                        local_file = f"{module_name.replace('.', '/')}.py"
                        if os.path.exists(local_file):
                            self.dependencies.add(local_file)
                            print(f"  {'  ' * depth}↳ {local_file}")
                            self.analyze_import_chain(local_file, depth + 1, max_depth)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        local_file = f"{module_name.replace('.', '/')}.py"
                        if os.path.exists(local_file):
                            self.dependencies.add(local_file)
                            print(f"  {'  ' * depth}↳ {local_file}")
                            self.analyze_import_chain(local_file, depth + 1, max_depth)
        
        except Exception as e:
            pass  # Skip files that can't be parsed
    
    def run_analysis(self):
        """Run complete dependency analysis"""
        print("🚀 FTMO TRADING SYSTEM DEPENDENCY ANALYSIS")
        print("=" * 60)
        
        # Analyze main file
        if not self.analyze_main_file():
            return
        
        # Find connected files
        connected_files = self.find_connected_files()
        
        # Analyze import chains for key files
        print(f"\n📂 ANALYZING IMPORT CHAINS...")
        for file in list(connected_files)[:5]:  # Limit depth for key files
            if os.path.exists(file):
                print(f"🔍 Analyzing: {file}")
                self.analyze_import_chain(file)
        
        # Display results
        print(f"\n📊 DEPENDENCY ANALYSIS RESULTS")
        print("=" * 40)
        print(f"Main file: {self.main_file}")
        print(f"Dependencies found: {len(self.dependencies)}")
        print(f"Connected files: {len(connected_files)}")
        
        if self.import_errors:
            print(f"\n❌ Import errors: {len(self.import_errors)}")
            for error in self.import_errors[:3]:  # Show first 3 errors
                print(f"  {error}")
        
        # Create dependency map
        self.create_dependency_map(connected_files)
    
    def create_dependency_map(self, connected_files):
        """Create a visual dependency map"""
        print(f"\n🗺️ DEPENDENCY MAP:")
        print("enhanced_trading_dashboard.py")
        
        for file in sorted(connected_files):
            print(f"  └── {file}")
            
            # Show sub-dependencies for key files
            if any(keyword in file for keyword in ['ensemble', 'integrate', 'professional']):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for notable imports in this file
                    notable_imports = []
                    for line in content.split('\n'):
                        if 'import' in line and any(keyword in line for keyword in ['ftmo_', 'enhanced_', 'advanced_']):
                            notable_imports.append(line.strip())
                    
                    for imp in notable_imports[:2]:  # Show first 2 notable imports
                        print(f"      └── {imp}")
                
                except:
                    pass

# Run analysis
if __name__ == "__main__":
    analyzer = SystemDependencyAnalyzer()
    analyzer.run_analysis()
