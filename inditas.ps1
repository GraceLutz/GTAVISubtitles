# Első futáskor telepít, utána csak indít. Paraméterek továbbadva, pl.:
#   .\inditas.ps1 --list-devices
#   .\inditas.ps1 --device 1 --windowed
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Első indítás: telepítés (pár perc)..."
    py -3 -m venv .venv
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
    # A rapidocr a GUI nélküli OpenCV-t is felteszi; ablakhoz a teljes kell
    & $python -m pip install --force-reinstall --no-deps opencv-python==5.1.0.95
}

& $python -m gta_felirat @args
