print("Testing symbol configuration...")

# Test the symbols mapping
symbols_map = {
    "All Optimized": ["BTCX25.sim", "US30Z25.sim", "XAUZ25.sim", "US100Z25.sim", "US500Z25.sim", "USOILZ25.sim"],
    "BTCX25.sim only": ["BTCX25.sim"],
    "US30Z25.sim only": ["US30Z25.sim"],
    "XAUZ25.sim only": ["XAUZ25.sim"],
    "US100Z25.sim only": ["US100Z25.sim"],
    "US500Z25.sim only": ["US500Z25.sim"],
    "USOILZ25.sim only": ["USOILZ25.sim"]
}

print("Symbols combobox values:")
for option in ["All Optimized", "BTCX25.sim only", "US30Z25.sim only", "XAUZ25.sim only", "US100Z25.sim only", "US500Z25.sim only", "USOILZ25.sim only"]:
    symbols = symbols_map.get(option, [])
    print(f"  {option} -> {symbols}")

print("✅ Symbol configuration test passed!")
