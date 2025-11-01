# FTMO Trading System - Directory Maintenance Script
param(
    [switch]$Cleanup,
    [switch]$Backup,
    [switch]$Verify
)

function Invoke-Backup {
    Write-Host "Creating backup..." -ForegroundColor Green
    $backupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $backupDir | Out-Null
    
    # Backup key files
    $filesToBackup = @("*.py", "*.json", "*.md", "*.txt", "*.log")
    foreach ($pattern in $filesToBackup) {
        Get-ChildItem -Path . -Filter $pattern -File | ForEach-Object {
            $target = Join-Path $backupDir $_.Name
            Copy-Item $_.FullName $target -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Backup created: $backupDir" -ForegroundColor Green
}

function Invoke-Cleanup {
    Write-Host "Cleaning up temporary files..." -ForegroundColor Yellow
    
    $patternsToRemove = @("*.tmp", "*.log.*", "*.bak", "__pycache__", "*.pyc")
    foreach ($pattern in $patternsToRemove) {
        Get-ChildItem -Path . -Filter $pattern -Recurse -ErrorAction SilentlyContinue | 
        ForEach-Object {
            try {
                if ($_.PSIsContainer) {
                    Remove-Item $_.FullName -Recurse -Force
                } else {
                    Remove-Item $_.FullName -Force
                }
                Write-Host "Removed: $($_.Name)" -ForegroundColor Red
            } catch {
                Write-Host "Couldn't remove: $($_.Name)" -ForegroundColor Yellow
            }
        }
    }
}

function Invoke-Verify {
    Write-Host "Verifying directory structure..." -ForegroundColor Cyan
    
    $requiredDirs = @("core", "data", "logs", "reports", "backup", "docs")
    $missingDirs = @()
    
    foreach ($dir in $requiredDirs) {
        if (Test-Path $dir) {
            Write-Host "✓ $dir" -ForegroundColor Green
        } else {
            Write-Host "✗ $dir" -ForegroundColor Red
            $missingDirs += $dir
        }
    }
    
    if ($missingDirs) {
        Write-Host "Missing directories: $($missingDirs -join ', ')" -ForegroundColor Yellow
    }
}

# Main execution
if ($Backup) {
    Invoke-Backup
}

if ($Cleanup) {
    Invoke-Cleanup
}

if ($Verify) {
    Invoke-Verify
}

if (-not ($Backup -or $Cleanup -or $Verify)) {
    Write-Host @"
FTMO Trading System - Directory Maintenance
===========================================

Usage:
    .\maintain.ps1 -Backup    # Create backup
    .\maintain.ps1 -Cleanup   # Clean temporary files
    .\maintain.ps1 -Verify    # Verify structure
    .\maintain.ps1 -Backup -Cleanup  # Both backup and cleanup

Recommended monthly maintenance:
    .\maintain.ps1 -Backup -Cleanup -Verify
"@
}
