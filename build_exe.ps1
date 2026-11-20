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
    throw "Nincs .venv. Előbb futtasd egyszer az .\inditas.ps1-et, az telepíti a függőségeket."
}
if ($PSScriptRoot.Length -gt $MaxSafePathLength) {
    Write-Warning "A projekt útvonala hosszú ($PSScriptRoot). Ha az exe nem tölti be az OCR-t, buildelj rövidebb helyről, pl. C:\gta_felirat."
}

& $python -m pip install --disable-pip-version-check -q "pyinstaller==$PyInstallerVersion"
if ($LASTEXITCODE -ne 0) { throw "A PyInstaller telepítése nem sikerült." }

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
    if ($LASTEXITCODE -ne 0) { throw "A PyInstaller build ($Mode) nem sikerült." }
}

Invoke-PyInstaller "--onedir" (Join-Path $buildDir "onedir")
Invoke-PyInstaller "--onefile" (Join-Path $buildDir "onefile")

# A mappás változatban a config.json eleve az exe mellett van (onnan olvas és oda ment a program)
Copy-Item $defaultConfig (Join-Path $appDir "config.json")
Copy-Item (Join-Path $PSScriptRoot "packaging\OLVASS_EL.txt") (Join-Path $appDir "OLVASS_EL.txt")

# Mindkét változat ugyanazokat a fájlokat csomagolja, ezért a mappás változaton ellenőrzünk
$leaked = Get-ChildItem $appDir -Recurse -File -Include $ForbiddenFiles
if ($leaked) {
    throw "Tiltott fájl került a csomagba: $($leaked.FullName -join ', ')"
}
$models = Get-ChildItem $appDir -Recurse -Filter "*.onnx"
$ortDll = Get-ChildItem $appDir -Recurse -Filter "onnxruntime.dll"
if (-not $models -or -not $ortDll) {
    throw "Hiányzik a RapidOCR-modell vagy az onnxruntime.dll a csomagból."
}

if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path $appDir -DestinationPath $zipPath -CompressionLevel Optimal

foreach ($artifact in @($oneFileExe, $zipPath)) {
    $sizeMb = [math]::Round((Get-Item $artifact).Length / 1MB, 1)
    Write-Host "Kész: $artifact ($sizeMb MB)"
}
