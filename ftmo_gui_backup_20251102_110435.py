# FTMO GUI Dashboard - Real-time Trading Monitor
import sys
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QProgressBar, QTableWidget, 
                             QTableWidgetItem, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from ftmo_minimal import FTMOMinimalSystem

class FTMO_GUI_Dashboard(QMainWindow):
    def __init__(self, trading_system):
        super().__init__()
        self.trading_system = trading_system
        self.setWindowTitle("FTMO Trading Dashboard - 200k Challenge")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Initialize GUI components
        self.init_ui()
        
        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(2000)  # Update every 2 seconds
        
        # Trading thread
        self.trading_thread = None
        self.trading_active = False
        
    def init_ui(self):
        """Initialize UI components"""
        # Header
        self.create_header()
        
        # Challenge progress section
        self.create_challenge_section()
        
        # Trading metrics section
        self.create_metrics_section()
        
        # Risk management section
        self.create_risk_section()
        
        # Recent trades section
        self.create_trades_section()
        
        # System log
        self.create_log_section()
        
    def create_header(self):
        """Create dashboard header"""
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("FTMO 200K CHALLENGE DASHBOARD")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2E86AB; padding: 10px;")
        
        header_layout.addWidget(title_label)
        self.main_layout.addLayout(header_layout)
        
    def create_challenge_section(self):
        """Create challenge progress section"""
        challenge_group = QGroupBox("Challenge Progress")
        challenge_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        
        # Progress labels
        progress_layout = QHBoxLayout()
        
        self.current_profit_label = QLabel("Current Profit: $0.00")
        self.target_label = QLabel("Target: $10,000.00")
        self.progress_percent_label = QLabel("0.0% Complete")
        
        progress_layout.addWidget(self.current_profit_label)
        progress_layout.addWidget(self.target_label)
        progress_layout.addWidget(self.progress_percent_label)
        
        challenge_layout.addWidget(self.progress_bar)
        challenge_layout.addLayout(progress_layout)
        challenge_group.setLayout(challenge_layout)
        self.main_layout.addWidget(challenge_group)
        
    def create_metrics_section(self):
        """Create trading metrics section"""
        metrics_group = QGroupBox("Trading Metrics")
        metrics_layout = QHBoxLayout()
        
        # Left column
        left_layout = QVBoxLayout()
        self.strategy_label = QLabel("Current Strategy: XGBoost")
        self.win_rate_label = QLabel("Win Rate: 0.0%")
        self.total_trades_label = QLabel("Total Trades: 0")
        
        left_layout.addWidget(self.strategy_label)
        left_layout.addWidget(self.win_rate_label)
        left_layout.addWidget(self.total_trades_label)
        
        # Right column
        right_layout = QVBoxLayout()
        self.daily_profit_label = QLabel("Daily P/L: $0.00")
        self.active_trades_label = QLabel("Active Trades: 0")
        self.signal_confidence_label = QLabel("Signal Confidence: 0.0%")
        
        right_layout.addWidget(self.daily_profit_label)
        right_layout.addWidget(self.active_trades_label)
        right_layout.addWidget(self.signal_confidence_label)
        
        metrics_layout.addLayout(left_layout)
        metrics_layout.addLayout(right_layout)
        metrics_group.setLayout(metrics_layout)
        self.main_layout.addWidget(metrics_group)
        
    def create_risk_section(self):
        """Create risk management section"""
        risk_group = QGroupBox("Risk Management")
        risk_layout = QHBoxLayout()
        
        # Risk indicators
        self.risk_level_label = QLabel("Risk Level: NORMAL")
        self.risk_level_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.drawdown_label = QLabel("Max Drawdown: $0.00")
        self.position_size_label = QLabel("Position Size: 2.0%")
        
        risk_layout.addWidget(self.risk_level_label)
        risk_layout.addWidget(self.drawdown_label)
        risk_layout.addWidget(self.position_size_label)
        
        risk_group.setLayout(risk_layout)
        self.main_layout.addWidget(risk_group)
        
    def create_trades_section(self):
        """Create recent trades table"""
        trades_group = QGroupBox("Recent Trades")
        trades_layout = QVBoxLayout()
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels(["Time", "Symbol", "Type", "Size", "P/L", "Status"])
        self.trades_table.setColumnWidth(0, 150)
        self.trades_table.setColumnWidth(1, 80)
        self.trades_table.setColumnWidth(2, 60)
        self.trades_table.setColumnWidth(3, 60)
        self.trades_table.setColumnWidth(4, 80)
        self.trades_table.setColumnWidth(5, 80)
        
        trades_layout.addWidget(self.trades_table)
        trades_group.setLayout(trades_layout)
        self.main_layout.addWidget(trades_group)
        
    def create_log_section(self):
        """Create system log section"""
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        self.main_layout.addWidget(log_group)
        
    def update_dashboard(self):
        """Update all dashboard elements"""
        try:
            status = self.trading_system.get_system_status()
            self.update_challenge_progress(status)
            self.update_trading_metrics(status)
            self.update_risk_management(status)
            self.update_recent_trades()
            self.update_system_log()
            
        except Exception as e:
            self.log_text.append(f"Error updating dashboard: {e}")
            
    def update_challenge_progress(self, status):
        """Update challenge progress section"""
        current_profit = status.get('progress', 0) * 10000 / 100  # Convert percentage to dollars
        progress_percent = status.get('progress', 0)
        
        self.progress_bar.setValue(int(progress_percent))
        self.current_profit_label.setText(f"Current Profit: ${current_profit:,.2f}")
        self.progress_percent_label.setText(f"{progress_percent:.1f}% Complete")
        
        # Update progress bar color based on progress
        if progress_percent >= 80:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        elif progress_percent >= 50:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #FFC107; }")
        else:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
            
    def update_trading_metrics(self, status):
        """Update trading metrics"""
        self.strategy_label.setText(f"Current Strategy: {status.get('strategy', 'XGBoost')}")
        
        # Simulate metrics (would come from actual trading data)
        win_rate = 65.5  # Would be calculated from trades
        total_trades = 12  # Would be from trade history
        daily_profit = 450.50  # Would be calculated
        active_trades = 2  # Would be from current positions
        confidence = 72.3  # Would be from strategy
        
        self.win_rate_label.setText(f"Win Rate: {win_rate}%")
        self.total_trades_label.setText(f"Total Trades: {total_trades}")
        self.daily_profit_label.setText(f"Daily P/L: ${daily_profit:,.2f}")
        self.active_trades_label.setText(f"Active Trades: {active_trades}")
        self.signal_confidence_label.setText(f"Signal Confidence: {confidence}%")
        
    def update_risk_management(self, status):
        """Update risk management section"""
        risk_level = status.get('risk_level', 'NORMAL')
        self.risk_level_label.setText(f"Risk Level: {risk_level}")
        
        # Update risk level color
        if risk_level == 'NORMAL':
            self.risk_level_label.setStyleSheet("color: green; font-weight: bold;")
        elif risk_level == 'CAUTION':
            self.risk_level_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.risk_level_label.setStyleSheet("color: red; font-weight: bold;")
            
        # Simulate risk metrics
        drawdown = 1250.75  # Would be calculated
        position_size = 2.0  # Current risk percentage
        
        self.drawdown_label.setText(f"Max Drawdown: ${drawdown:,.2f}")
        self.position_size_label.setText(f"Position Size: {position_size}%")
        
    def update_recent_trades(self):
        """Update recent trades table"""
        # Sample trade data (would come from actual trading)
        sample_trades = [
            ["14:30:05", "US30", "BUY", "0.5", "+$125.50", "WIN"],
            ["14:45:22", "US100", "SELL", "0.3", "-$45.25", "LOSS"],
            ["15:10:18", "UAX", "BUY", "0.4", "+$87.30", "WIN"]
        ]
        
        self.trades_table.setRowCount(len(sample_trades))
        
        for row, trade in enumerate(sample_trades):
            for col, value in enumerate(trade):
                item = QTableWidgetItem(str(value))
                
                # Color code profit/loss
                if col == 4:  # P/L column
                    if value.startswith('+'):
                        item.setForeground(QColor(0, 128, 0))  # Green for profit
                    elif value.startswith('-'):
                        item.setForeground(QColor(255, 0, 0))  # Red for loss
                        
                # Color code status
                if col == 5:  # Status column
                    if value == "WIN":
                        item.setForeground(QColor(0, 128, 0))
                    else:
                        item.setForeground(QColor(255, 0, 0))
                        
                self.trades_table.setItem(row, col, item)
                
    def update_system_log(self):
        """Update system log with recent activity"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Sample log entries (would come from actual system)
        log_entries = [
            f"{current_time} - Market analysis complete",
            f"{current_time} - XGBoost signal generated: BUY US30 (72% confidence)",
            f"{current_time} - Risk validation passed",
            f"{current_time} - Trade executed: BUY US30 0.5 lots"
        ]
        
        # Keep only last 10 entries
        current_text = self.log_text.toPlainText()
        lines = current_text.split('\n')[-8:]  # Keep last 8 lines
        
        # Add new entry if needed
        if not lines or log_entries[0] not in lines[-1]:
            self.log_text.append(log_entries[0])
            
    def start_trading(self):
        """Start trading in background thread"""
        if not self.trading_active:
            self.trading_active = True
            self.trading_thread = threading.Thread(target=self.run_trading)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            self.log_text.append("Trading started...")
            
    def run_trading(self):
        """Run trading in background thread"""
        try:
            # This would start the actual trading system
            # For now, simulate trading activity
            while self.trading_active:
                time.sleep(5)
        except Exception as e:
            self.log_text.append(f"Trading error: {e}")
            
    def stop_trading(self):
        """Stop trading"""
        self.trading_active = False
        self.log_text.append("Trading stopped...")
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_trading()
        event.accept()

def launch_gui():
    """Launch the GUI dashboard"""
    # Initialize trading system
    trading_system = FTMOMinimalSystem("1600038177", "200k", live_mode=False)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show dashboard
    dashboard = FTMO_GUI_Dashboard(trading_system)
    dashboard.show()
    
    # Start trading
    dashboard.start_trading()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    launch_gui()
