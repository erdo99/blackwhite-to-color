# export_venv_info.ps1 ile uretilen yedek dosyalarini listeler.
# Kullanim:
#   .\scripts\list_venv_backups.ps1
#   .\scripts\list_venv_backups.ps1 -ScanRoot "C:\Users\erdem"

param(
    [string]$ScanRoot = $env:USERPROFILE
)

if (-not (Test-Path $ScanRoot)) {
    Write-Error "Klasor yok: $ScanRoot"
    exit 1
}

Write-Host "Taranan kok: $ScanRoot`n" -ForegroundColor Cyan

$pattern = "python_venv_backup__*"
$jsonFiles = Get-ChildItem -Path $ScanRoot -Recurse -Filter "$pattern.json" -File -ErrorAction SilentlyContinue
$mdFiles   = Get-ChildItem -Path $ScanRoot -Recurse -Filter "$pattern.md"   -File -ErrorAction SilentlyContinue
$combined  = Join-Path $ScanRoot "all_python_venvs_inventory.json"

Write-Host "=== JSON yedekleri ($($jsonFiles.Count) adet) ===" -ForegroundColor Yellow
$jsonFiles | Sort-Object FullName | ForEach-Object {
    "{0,-12} {1}" -f ("{0:N0} B" -f $_.Length), $_.FullName
}

Write-Host "`n=== MD yedekleri ($($mdFiles.Count) adet) ===" -ForegroundColor Yellow
$mdFiles | Sort-Object FullName | ForEach-Object {
    "{0,-12} {1}" -f ("{0:N0} B" -f $_.Length), $_.FullName
}

Write-Host "`n=== Toplu envanter ===" -ForegroundColor Yellow
if (Test-Path $combined) {
    $c = Get-Item $combined
    "{0,-12} {1}" -f ("{0:N0} B" -f $c.Length), $c.FullName
} else {
    "(yok) $combined"
}

Write-Host "`nOz: $($jsonFiles.Count) json, $($mdFiles.Count) md" -ForegroundColor Green
