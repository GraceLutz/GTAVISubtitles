# Telepítés nélkül futtatható Windows-csomagok (PyInstaller) streamereknek.
#   .\build_exe.ps1
# Eredmény:
#   dist\GTA6Felirat.exe            egyetlen fájl: letöltés után dupla kattintással indul
#   dist\GTA6Felirat\ + dist\GTA6Felirat-win64.zip   mappás változat (gyorsabb indulás)
# Előfeltétel: a .venv már létezik (egyszer futtasd az inditas.ps1-et).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$AppName = "GTA6Felirat"
$PyInstallerVersion = "6.22.3"
# Hosszú útvonalon az onnxruntime DLL-betöltése elromlik; rövid helyen (pl. C:\gta_felirat) buildelj
$MaxSafePathLength = 40
# Ezek sosem kerülhetnek a csomagba (mentett fordítások, titkok)
$ForbiddenFiles = @("mentett_forditasok.json", ".env", "*.tmp")

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No .venv found. Run .\inditas.ps1 once first; it installs the dependencies."
}
if ($PSScriptRoot.Length -gt $MaxSafePathLength) {
    Write-Warning "The project path is long ($PSScriptRoot). If the exe cannot load the OCR, build from a shorter path, e.g. C:\gta_felirat."
}

& $python -m pip install --disable-pip-version-check -q "pyinstaller==$PyInstallerVersion"
if ($LASTEXITCODE -ne 0) { throw "Installing PyInstaller failed." }

$distDir = Join-Path $PSScriptRoot "dist"
$buildDir = Join-Path $PSScriptRoot "build"
$appDir = Join-Path $distDir $AppName
$oneFileExe = Join-Path $distDir "$AppName.exe"
$zipPath = Join-Path $distDir "$AppName-win64.zip"
$launcher = Join-Path $PSScriptRoot "packaging\exe_launcher.py"
New-Item -ItemType Directory -Force $buildDir | Out-Null

# Alap config.json a csomagba: a commitolt változat, nem a helyi, játék közben átállított beállítások
$defaultConfig = Join-Path $buildDir "default_config.json"
$committedConfig = git show HEAD:config.json 2>$null
if ($LASTEXITCODE -eq 0 -and $committedConfig) {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($defaultConfig, (($committedConfig -join "`n") + "`n"), $utf8NoBom)
} else {
    Copy-Item (Join-Path $PSScriptRoot "config.json") $defaultConfig
}

function Invoke-PyInstaller([string]$Mode, [string]$WorkDir) {
    # --collect-data rapidocr: ONNX-modellek és yaml-konfigok; --collect-binaries onnxruntime: DLL-ek
    & $python -m PyInstaller --noconfirm --clean $Mode --console `
        --name $AppName `
        --distpath $distDir --workpath $WorkDir --specpath $WorkDir `
        --paths $PSScriptRoot `
        --add-data "$defaultConfig;." `
        --collect-data rapidocr `
        --collect-binaries onnxruntime `
        --exclude-module pytest `
        $launcher
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build ($Mode) failed." }
}

Invoke-PyInstaller "--onedir" (Join-Path $buildDir "onedir")
Invoke-PyInstaller "--onefile" (Join-Path $buildDir "onefile")

# A mappás változatban a config.json eleve az exe mellett van (onnan olvas és oda ment a program)
Copy-Item $defaultConfig (Join-Path $appDir "config.json")
Copy-Item (Join-Path $PSScriptRoot "packaging\README.txt") (Join-Path $appDir "README.txt")

# Mindkét változat ugyanazokat a fájlokat csomagolja, ezért a mappás változaton ellenőrzünk
$leaked = Get-ChildItem $appDir -Recurse -File -Include $ForbiddenFiles
if ($leaked) {
    throw "A forbidden file ended up in the package: $($leaked.FullName -join ', ')"
}
$models = Get-ChildItem $appDir -Recurse -Filter "*.onnx"
$ortDll = Get-ChildItem $appDir -Recurse -Filter "onnxruntime.dll"
if (-not $models -or -not $ortDll) {
    throw "The RapidOCR model or onnxruntime.dll is missing from the package."
}

if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path $appDir -DestinationPath $zipPath -CompressionLevel Optimal

foreach ($artifact in @($oneFileExe, $zipPath)) {
    $sizeMb = [math]::Round((Get-Item $artifact).Length / 1MB, 1)
    Write-Host "Done: $artifact ($sizeMb MB)"
}
