# Sanal ortamlari silmeden once: pyvenv.cfg, Python surumu, pip paketleri (surumlerle) yedekler.
#
# Dosyalarin yeri (varsayilan):
#   Her venv'in BIR UST klasorune (proje kokune) yazar, ornek:
#   C:\Users\erdem\BoT-SORT\botsort_env  ->  C:\Users\erdem\BoT-SORT\python_venv_backup__botsort_env.json
#   Boylece venv klasorunu sildiginde yedek dosyalar KALIR.
#
# Venv icine yazmak istersen (silince gider): -WriteInsideVenv
#
# Kullanim:
#   .\scripts\export_venv_info.ps1 -ScanRoot "C:\Users\erdem"
#   .\scripts\export_venv_info.ps1 -ScanRoot "C:\Users\erdem" -SkipCombined
#   .\scripts\export_venv_info.ps1 -ScanRoot "C:\Users\erdem" -NoPip

param(
    [string]$ScanRoot = $env:USERPROFILE,
    [switch]$SkipCombined,
    [switch]$JsonOnly,
    [switch]$MdOnly,
    [switch]$WriteInsideVenv,
    [switch]$NoPip
)

$ErrorActionPreference = "Continue"
$all = [System.Collections.ArrayList]::new()
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"

function Get-SafeLeafName {
    param([string]$Path)
    $leaf = Split-Path $Path -Leaf
    ($leaf -replace '[^\w\-\.]', '_')
}

Get-ChildItem -Path $ScanRoot -Recurse -Filter "pyvenv.cfg" -File -ErrorAction SilentlyContinue | ForEach-Object {
    $cfgFile = $_.FullName
    $venvDir = $_.Directory.FullName
    $parent = Split-Path $venvDir -Parent
    $leafSafe = Get-SafeLeafName $venvDir

    $cfgText = Get-Content -Path $cfgFile -Raw -ErrorAction SilentlyContinue
    $pyExe = Join-Path $venvDir "Scripts\python.exe"

    $versionLine = $null
    $runError = $null
    if (Test-Path $pyExe) {
        try {
            $versionLine = (& $pyExe --version 2>&1 | Out-String).Trim()
        } catch {
            $runError = $_.Exception.Message
        }
        if (-not $versionLine) {
            try {
                $versionLine = (& $pyExe -c "import sys; print(sys.version)" 2>&1 | Out-String).Trim()
            } catch {
                $runError = "$runError; $($_.Exception.Message)"
            }
        }
    } else {
        $runError = "Scripts\python.exe bulunamadi"
    }

    if (-not $versionLine -and $cfgText -match '(?m)^version\s*=\s*(.+)$') {
        $versionLine = "pyvenv.cfg version satiri: $($Matches[1].Trim())"
    }

    $pythonOk = [bool](
        $versionLine -and
        ($versionLine -notmatch "did not find executable")
    )

    $pipFreezeText = $null
    $pipPackages = $null
    $pipError = $null
    if (-not $NoPip -and (Test-Path $pyExe) -and $pythonOk) {
        try {
            $pipFreezeText = (& $pyExe -m pip freeze 2>&1 | Out-String).Trim()
            if ($pipFreezeText -match "Error|error:") { throw $pipFreezeText }
        } catch {
            $pipError = "pip freeze: $($_.Exception.Message)"
            $pipFreezeText = $null
        }
        try {
            $jsonOut = & $pyExe -m pip list --format=json 2>&1 | Out-String
            if ($jsonOut -match "^\s*\[") {
                $pipPackages = $jsonOut | ConvertFrom-Json
            } else {
                throw $jsonOut
            }
        } catch {
            if (-not $pipError) { $pipError = "pip list: $($_.Exception.Message)" }
        }
    } elseif ($NoPip) {
        $pipError = "NoPip: atlandi"
    } else {
        $pipError = "Python calismiyor veya yok; pip alinamadi"
    }

    $outputNote = if ($WriteInsideVenv) {
        "Dosyalar venv icinde; venv silinirse bu json/md de silinir."
    } else {
        "Dosyalar venv'in UST proje klasorunde; venv silinsen bu yedekler kalir."
    }

    $entry = [ordered]@{
        exportedAt       = $timestamp
        scanRoot         = $ScanRoot
        venvPath         = $venvDir
        parentFolder     = $parent
        outputNote       = $outputNote
        pyvenvCfgPath    = $cfgFile
        pyvenvCfgRaw     = $cfgText
        pythonExe        = $pyExe
        pythonWorks      = $pythonOk
        versionOrOutput  = $versionLine
        error            = $runError
        pipFreeze        = $pipFreezeText
        pipPackages      = $pipPackages
        pipError         = $pipError
    }

    [void]$all.Add([pscustomobject]$entry)

    $baseName = "python_venv_backup__$leafSafe"
    $outDir = if ($WriteInsideVenv) { $venvDir } else { $parent }

    if (-not $MdOnly) {
        $jsonPath = Join-Path $outDir "$baseName.json"
        $entry | ConvertTo-Json -Depth 12 | Set-Content -Path $jsonPath -Encoding UTF8
        Write-Host "JSON: $jsonPath"
    }
    if (-not $JsonOnly) {
        $mdPath = Join-Path $outDir "$baseName.md"
        $verShow = if ($versionLine) { $versionLine } else { '—' }
        $errShow = if ($runError) { $runError } else { '—' }
        $pipErrShow = if ($pipError) { $pipError } else { '—' }
        $fence = '```'
        $freezeBlock = if ($pipFreezeText) { $pipFreezeText } else { '(yok veya alinamadi)' }
        $pkgTable = ""
        if ($pipPackages -and $pipPackages.Count -gt 0) {
            $pkgTable = "| Paket | Surum |`n|-------|-------|`n"
            foreach ($p in $pipPackages) {
                $n = $p.name
                $v = $p.version
                $pkgTable += "| $n | $v |`n"
            }
        } else {
            $pkgTable = "_pip list JSON alinamadi veya bos._`n"
        }
        $md = @"
# Python venv yedek bilgisi

- **Tarih:** $timestamp
- **Venv klasoru:** ``$venvDir``
- **Dosya yazilan klasor:** ``$outDir``
- **Not:** $outputNote

## pyvenv.cfg

$fence
$cfgText
$fence

## Python

- **python.exe:** ``$pyExe``
- **Surum / cikti:** $verShow
- **Calisiyor:** $pythonOk
- **Hata (varsa):** $errShow

## Pip — paket listesi (tablo)

$pkgTable

## Pip freeze (requirements benzeri)

$fence
$freezeBlock
$fence

## Pip ek hata

$pipErrShow

"@
        Set-Content -Path $mdPath -Value $md -Encoding UTF8
        Write-Host "MD:   $mdPath"
    }
}

if (-not $SkipCombined) {
    $combined = Join-Path $ScanRoot "all_python_venvs_inventory.json"
    $all | ConvertTo-Json -Depth 14 | Set-Content -Path $combined -Encoding UTF8
    Write-Host ""
    Write-Host "Toplu liste: $combined"
}

Write-Host ""
Write-Host "Toplam $($all.Count) venv islendi."
