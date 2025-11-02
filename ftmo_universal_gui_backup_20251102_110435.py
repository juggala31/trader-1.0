# FTMO Universal GUI Launcher - Tries PyQt5 first, falls back to tkinter
import sys
import os

def try_pyqt5_gui():
    """Try to launch PyQt5 GUI"""
    try:
        from PyQt5.QtWidgets import QApplication
        from ftmo_gui import launch_gui
        print("🎯 Launching PyQt5 GUI Dashboard...")
        launch_gui()
        return True
    except ImportError:
        print("❌ PyQt5 not available")
        return False
    except Exception as e:
        print(f"❌ PyQt5 GUI error: {e}")
        return False

def launch_tkinter_gui():
    """Launch tkinter GUI (fallback)"""
    try:
        from ftmo_tkinter_gui import main as tkinter_main
        print("🎯 Launching Tkinter GUI Dashboard...")
        tkinter_main()
        return True
    except Exception as e:
        print(f"❌ Tkinter GUI error: {e}")
        return False

def install_pyqt5():
    """Attempt to install PyQt5"""
    print("🔧 Attempting to install PyQt5...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
        print("✅ PyQt5 installed successfully!")
        return True
    except Exception as e:
        print(f"❌ PyQt5 installation failed: {e}")
        return False

def main():
    print("🚀 FTMO UNIVERSAL GUI LAUNCHER")
    print("==============================")
    print("This will attempt to launch the best available GUI")
    print("==============================")
    
    # First try PyQt5 GUI
    if try_pyqt5_gui():
        return
        
    # If PyQt5 fails, offer to install it
    print("\nPyQt5 GUI is not available.")
    response = input("Would you like to install PyQt5? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        if install_pyqt5():
            # Try PyQt5 again after installation
            if try_pyqt5_gui():
                return
                
    # Fall back to tkinter GUI
    print("\nFalling back to Tkinter GUI (no installation required)...")
    if launch_tkinter_gui():
        return
        
    # If all GUI options fail, start terminal interface
    print("\n❌ All GUI options failed. Starting terminal interface...")
    from ftmo_minimal import FTMOMinimalSystem
    system = FTMOMinimalSystem()
    system.start_trading()

if __name__ == "__main__":
    main()
