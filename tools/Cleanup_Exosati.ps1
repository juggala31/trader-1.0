param(
    # Root of your project
    [string]$Root = "C:\ExosatiTraderPy",

    # Also back up and remove .git (off by default)
    [switch]$ArchiveGit = $false,

    # Move big CSVs in /data to backup (keeps small samples)
    [switch]$MoveDataCSVs = $true,

    # Move heavy trained models to backup (keeps code)
    [switch]$MoveModels = $true,

    # Keep CSV files <= this size (KB) to retain tiny samples
    [int]$CsvSizeKeepKB = 250
)

# ---------------------------
# Helpers
# ---------------------------
function New-Dir([string]$p){ if(-not (Test-Path $p)){ New-Item -ItemType Directory -Path $p | Out-Null } }

function Relative-Path([string]$full, [string]$root){
    return $full.Substring($root.Length).TrimStart('\','/')
}

function SafeMove([System.IO.FileSystemInfo]$item, [string]$root, [string]$backupRemoved){
    try{
        $rel = Relative-Path $item.FullName $root
        $dest = Join-Path $backupRemoved $rel
        $destDir = Split-Path $dest -Parent
        New-Dir $destDir
        Move-Item -LiteralPath $item.FullName -Destination $dest -Force
        return $true
    } catch {
        Write-Warning "Failed to move: $($item.FullName) -> $($_.Exception.Message)"
        return $false
    }
}

function Add-PathIfExists([string]$p, [ref]$bag){
    if(Test-Path $p){ $bag.Value += ,(Get-Item -LiteralPath $p) }
}

# ---------------------------
# Prep
# ---------------------------
if(-not (Test-Path $Root)){ throw "Root not found: $Root" }
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$backupRoot = Join-Path $Root "_cleanup_backup"
$backupDir  = Join-Path $backupRoot ("Backup_{0}" -f $stamp)
$backupRemoved = Join-Path $backupDir "Removed"
$backupMeta    = Join-Path $backupDir "Meta"
$zipPath       = Join-Path $backupRoot ("Backup_{0}.zip" -f $stamp)

New-Dir $backupRoot
New-Dir $backupDir
New-Dir $backupRemoved
New-Dir $backupMeta

$logFile = Join-Path $backupMeta "moved_items_$stamp.txt"
$moved = New-Object System.Collections.Generic.List[string]

Write-Host "[1/6] Scanning..." -ForegroundColor Cyan

# ---------------------------
# Collect items to move
# ---------------------------
$toMove = @()

# 1) Big root items
Add-PathIfExists (Join-Path $Root ".venv") ([ref]$toMove)
if($ArchiveGit){ Add-PathIfExists (Join-Path $Root ".git") ([ref]$toMove) }

# 2) __pycache__ dirs
$toMove += Get-ChildItem -Path $Root -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue

# 3) *.pyc files
$toMove += Get-ChildItem -Path $Root -Recurse -Force -Include *.pyc -File -ErrorAction SilentlyContinue

# 4) backup files & backup folders
$toMove += Get-ChildItem -Path $Root -Recurse -Force -Include "*_backup.py","*_backup_*.py" -File -ErrorAction SilentlyContinue
$toMove += Get-ChildItem -Path $Root -Recurse -Force -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "backup_ai_*" }

# 5) Diagnostic / listing artifacts
$toMove += Get-ChildItem -Path $Root -Recurse -Force -Include "dir.txt","project_structure_report.txt","project_layout.txt","*report*.txt","*analysis*.txt" -File -ErrorAction SilentlyContinue

# 6) Models (heavy payloads)
if($MoveModels){
    $xgbDir = Join-Path $Root "models\xgb"
    if(Test-Path $xgbDir){
        $toMove += Get-ChildItem -Path $xgbDir -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "xgb_20*" }
    }
    # top-level *.pkl snapshots
    $toMove += Get-ChildItem -Path $Root -Force -Include *.pkl -File -ErrorAction SilentlyContinue
}

# 7) CSV data (move large ones only, keep small samples)
if($MoveDataCSVs){
    $dataDir = Join-Path $Root "data"
    if(Test-Path $dataDir){
        $csvs = Get-ChildItem -Path $dataDir -Recurse -Force -Include *.csv -File -ErrorAction SilentlyContinue
        foreach($f in $csvs){
            if([math]::Round($f.Length/1KB) -gt $CsvSizeKeepKB){
                $toMove += ,$f
            }
        }
    }
}

# De-duplicate while keeping objects
$toMove = $toMove | Sort-Object FullName -Unique

Write-Host "[2/6] Moving items into backup..." -ForegroundColor Cyan
$ok = 0
foreach($it in $toMove){
    if(SafeMove $it $Root $backupRemoved){
        $ok++
        $moved.Add($it.FullName)
    }
}
$moved | Out-File -FilePath $logFile -Encoding UTF8

Write-Host ("Moved {0} items to backup." -f $ok) -ForegroundColor Green

# ---------------------------
# Ensure clean structure exists
# ---------------------------
Write-Host "[3/6] Ensuring clean folder structure..." -ForegroundColor Cyan

$dirsToEnsure = @(
    (Join-Path $Root "models"),
    (Join-Path $Root "models\xgb"),
    (Join-Path $Root "models\ensemble"),
    (Join-Path $Root "data"),
    (Join-Path $Root "tools"),
    (Join-Path $Root "scripts"),
    (Join-Path $Root "logs")
)
foreach($d in $dirsToEnsure){ New-Dir $d }

# Keep a placeholder in empty dirs for git friendliness
$placeholders = @()
foreach($d in $dirsToEnsure){
    $gitkeep = Join-Path $d ".gitkeep"
    if(-not (Test-Path $gitkeep)){ New-Item -ItemType File -Path $gitkeep -Force | Out-Null; $placeholders += $gitkeep }
}

# ---------------------------
# .gitignore (idempotent write)
# ---------------------------
Write-Host "[4/6] Writing .gitignore..." -ForegroundColor Cyan
$gitignorePath = Join-Path $Root ".gitignore"
$gitignore = @"
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.build/

# Environments
.venv/
.env/
.env.*

# Data & Models (keep code, ignore heavy artifacts)
data/**/*.csv
!data/*.sample.csv
models/**/xgb_20*/        # trained model drops
*.pkl

# Logs & temp
logs/
*.log

# OS / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

# Cleanup backups
_cleanup_backup/
"@

[System.IO.File]::WriteAllText($gitignorePath, $gitignore, $utf8NoBOM)

# ---------------------------
# Summary note in root
# ---------------------------
Write-Host "[5/6] Writing cleanup summary..." -ForegroundColor Cyan
$summary = @"
CLEANUP SUMMARY  ($stamp)

Backup directory:
  $backupDir

A zip of the backup was created (see below). You can restore anything from:
  $backupRemoved
(relative paths preserved)

Items moved: $ok
Log of moved items:
  $logFile

Parameters:
  ArchiveGit    = $ArchiveGit
  MoveDataCSVs  = $MoveDataCSVs (kept CSVs <= ${CsvSizeKeepKB}KB)
  MoveModels    = $MoveModels

Base folders ensured:
  models\\, models\\xgb\\, models\\ensemble\\, data\\, tools\\, scripts\\, logs\\

"@
$summaryPath = Join-Path $Root "CLEANUP_SUMMARY_$stamp.txt"
[System.IO.File]::WriteAllText($summaryPath, $summary, $utf8NoBOM)

# ---------------------------
# Zip the backup for safekeeping
# ---------------------------
Write-Host "[6/6] Creating backup zip..." -ForegroundColor Cyan
if(Test-Path $zipPath){ Remove-Item $zipPath -Force -ErrorAction SilentlyContinue }
Compress-Archive -Path (Join-Path $backupDir "*") -DestinationPath $zipPath -Force

Write-Host ""
Write-Host "✅ Done." -ForegroundColor Green
Write-Host "Backup folder : $backupDir"
Write-Host "Backup zip    : $zipPath"
Write-Host "Summary       : $summaryPath"
Write-Host ""
Write-Host "Tip: If you want to keep .git next time, omit -ArchiveGit." -ForegroundColor Yellow