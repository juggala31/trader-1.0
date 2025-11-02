# AUTO-GENERATED CLEANUP SCRIPT
# Generated: 11/01/2025 20:11:07
# Use with caution - review before running!

Write-Host "🧹 PROJECT CLEANUP SCRIPT" -ForegroundColor Cyan
Write-Host "=========================="

$backupDir = "cleanup_backup_20251101_201107"
New-Item -ItemType Directory -Path $backupDir | Out-Null
Write-Host "Backup directory created: $backupDir" -ForegroundColor Yellow

# Backup Directories
Write-Host 'Processing Backup Directories...' -ForegroundColor Cyan
# Remove directory: backup_ai_20251031_212252
# Move-Item -Path 'backup_ai_20251031_212252' -Destination $backupDir -Force

# Duplicate Files
Write-Host 'Processing Duplicate Files...' -ForegroundColor Cyan
# Review duplicate: backup_ai_20251031_212252
# Consider consolidating with similar files
# Review duplicate: backup_ai_20251031_212252
# Consider consolidating with similar files

# Test/Simple Versions
Write-Host 'Processing Test/Simple Versions...' -ForegroundColor Cyan
# Remove file: ftmo_minimal.py
# Move-Item -Path 'ftmo_minimal.py' -Destination $backupDir -Force
# Remove file: simple_regime_detector.py
# Move-Item -Path 'simple_regime_detector.py' -Destination $backupDir -Force
# Remove file: simple_test.py
# Move-Item -Path 'simple_test.py' -Destination $backupDir -Force
# Remove file: test_backtesting.py
# Move-Item -Path 'test_backtesting.py' -Destination $backupDir -Force
# Remove file: test_complete_system.py
# Move-Item -Path 'test_complete_system.py' -Destination $backupDir -Force
# Remove file: test_enhanced_dashboard.py
# Move-Item -Path 'test_enhanced_dashboard.py' -Destination $backupDir -Force
# Remove file: test_final_system.py
# Move-Item -Path 'test_final_system.py' -Destination $backupDir -Force
# Remove file: test_fixed_backtester.py
# Move-Item -Path 'test_fixed_backtester.py' -Destination $backupDir -Force
# Remove file: test_improved_backtester.py
# Move-Item -Path 'test_improved_backtester.py' -Destination $backupDir -Force
# Remove file: test_mt5_connection.py
# Move-Item -Path 'test_mt5_connection.py' -Destination $backupDir -Force
# Remove file: test_optimization.py
# Move-Item -Path 'test_optimization.py' -Destination $backupDir -Force
# Remove file: test_optimized_rl.py
# Move-Item -Path 'test_optimized_rl.py' -Destination $backupDir -Force
# Remove file: test_real_time_system.py
# Move-Item -Path 'test_real_time_system.py' -Destination $backupDir -Force
# Remove file: test_regime_detection.py
# Move-Item -Path 'test_regime_detection.py' -Destination $backupDir -Force
# Remove file: test_rl_system.py
# Move-Item -Path 'test_rl_system.py' -Destination $backupDir -Force
# Remove file: test_simple_system.py
# Move-Item -Path 'test_simple_system.py' -Destination $backupDir -Force
# Remove file: test_ultimate_dashboard.py
# Move-Item -Path 'test_ultimate_dashboard.py' -Destination $backupDir -Force
# Remove file: test_working_backtester.py
# Move-Item -Path 'test_working_backtester.py' -Destination $backupDir -Force
# Remove file: models\ensemble\simple_ensemble.py
# Move-Item -Path 'models\ensemble\simple_ensemble.py' -Destination $backupDir -Force

Write-Host "
✅ Cleanup script generated!" -ForegroundColor Green
Write-Host "📝 Review the script above before running" -ForegroundColor Yellow
Write-Host "🔒 Commented out for safety - remove '#' to execute" -ForegroundColor Red
