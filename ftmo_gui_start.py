# FTMO GUI Startup - Integrated Dashboard
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox

def check_dependencies():
    """Check if GUI dependencies are installed"""
    try:
        from PyQt5.QtWidgets import QMainWindow
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required GUI dependencies"""
    print("Installing GUI dependencies...")
    os.system("pip install PyQt5")
    print("Dependencies installed!")

def main():
    print("🚀 FTMO GUI DASHBOARD")
    print("=====================")
    
    # Check dependencies
    if not check_dependencies():
        print("PyQt5 not found. Installing...")
        install_dependencies()
        
    # Check if dependencies are now available
    if not check_dependencies():
        print("Failed to install PyQt5. Using terminal interface.")
        from ftmo_minimal import FTMOMinimalSystem
        system = FTMOMinimalSystem()
        system.start_trading()
        return
        
    # Launch GUI
    try:
        from ftmo_gui import launch_gui
        launch_gui()
    except Exception as e:
        print(f"GUI launch failed: {e}")
        print("Falling back to terminal interface...")
        from ftmo_minimal import FTMOMinimalSystem
        system = FTMOMinimalSystem()
        system.start_trading()

if __name__ == "__main__":
    main()
