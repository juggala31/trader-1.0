import re

print("=== SEARCHING FOR SYMBOLS COMBOBOX ===")

with open("enhanced_trading_dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if "values=[" in line and ("All Optimized" in line or "BTCX25" in line or "US30" in line):
        print(f"Line {i}: {line.strip()}")
        
print("")
print("If you see a line with 'US30 only' and 'XAU only', that's the problem!")
print("It should show 'XAUZ25.sim only' and 'US100Z25.sim only' instead.")
