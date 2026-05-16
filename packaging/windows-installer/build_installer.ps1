Param(
  [string]$ProjectRoot = "",
  [string]$OutputRoot = "",
  [string]$Version = "0.1.0",
  [switch]$SkipInstaller,
  [switch]$SkipGuiLaunch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-FullPath {
  Param([Parameter(Mandatory = $true)][string]$Path)
  return [System.IO.Path]::GetFullPath($Path)
}

function Assert-PathUnder {
  Param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Root
  )
  $fullPath = Resolve-FullPath $Path
  $fullRoot = (Resolve-FullPath $Root).TrimEnd('\', '/')
  if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to mutate path outside output root. Path=$fullPath Root=$fullRoot"
  }
}

function Reset-GeneratedDir {
  Param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$OutputRoot
  )
  $fullPath = Resolve-FullPath $Path
  Assert-PathUnder -Path $fullPath -Root $OutputRoot
  if (Test-Path -LiteralPath $fullPath) {
    Remove-Item -LiteralPath $fullPath -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
}

function Find-InnoCompiler {
  $cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
  if ($cmd) {
    return [string]$cmd.Path
  }

  $candidates = @()
  foreach ($root in @(${env:ProgramFiles(x86)}, $env:ProgramFiles, $env:LOCALAPPDATA)) {
    if (-not $root) {
      continue
    }
    if ($root -eq $env:LOCALAPPDATA) {
      $candidates += Join-Path $root "Programs\Inno Setup 6\ISCC.exe"
    } else {
      $candidates += Join-Path $root "Inno Setup 6\ISCC.exe"
    }
  }
  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
      return [string]$candidate
    }
  }
  return ""
}

function Ensure-InnoCompiler {
  $iscc = Find-InnoCompiler
  if ($iscc) {
    return $iscc
  }

  $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
  if (-not $winget) {
    throw "Inno Setup compiler was not found, and winget.exe is unavailable."
  }

  Write-Host "Inno Setup compiler not found. Installing with winget..."
  & $winget.Source install -e --id JRSoftware.InnoSetup --silent --accept-package-agreements --accept-source-agreements | Out-Host
  if ($LASTEXITCODE -ne 0) {
    throw "winget failed to install Inno Setup. ExitCode=$LASTEXITCODE"
  }

  $iscc = Find-InnoCompiler
  if (-not $iscc) {
    throw "Inno Setup installation completed, but ISCC.exe was still not found."
  }
  return $iscc
}

function Invoke-SmokeTestScript {
  Param(
    [Parameter(Mandatory = $true)][string]$SmokeScript,
    [Parameter(Mandatory = $true)][string]$PackageDir,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [switch]$SkipGuiLaunch
  )

  $args = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $SmokeScript,
    "-PackageDir",
    $PackageDir,
    "-OutputRoot",
    $OutputRoot
  )
  if ($SkipGuiLaunch) {
    $args += "-SkipGuiLaunch"
  }

  & powershell.exe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed for package directory: $PackageDir"
  }
}

function Wait-InstalledPackage {
  Param(
    [Parameter(Mandatory = $true)][string]$InstalledDir,
    [Parameter(Mandatory = $true)][string]$InstallLog,
    [int]$TimeoutSeconds = 180
  )

  $mainExe = Join-Path $InstalledDir "AnomalyDetectionRepair.exe"
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  do {
    $mainReady = Test-Path -LiteralPath $mainExe -PathType Leaf
    $logText = ""
    if (Test-Path -LiteralPath $InstallLog -PathType Leaf) {
      try {
        $logText = Get-Content -LiteralPath $InstallLog -Raw -Encoding UTF8
      } catch {
        $logText = ""
      }
    }
    if ($mainReady -and ($logText -match "Installation process succeeded")) {
      return
    }
    Start-Sleep -Seconds 1
  } while ((Get-Date) -lt $deadline)

  throw "Installed package was not ready after $TimeoutSeconds seconds. MainExe=$mainExe InstallLog=$InstallLog"
}

if (-not $ProjectRoot) {
  $ProjectRoot = Resolve-FullPath (Join-Path $PSScriptRoot "..\..")
} else {
  $ProjectRoot = Resolve-FullPath $ProjectRoot
}
if (-not $OutputRoot) {
  $OutputRoot = Join-Path $ProjectRoot "outputs\windows-installer"
}
$OutputRoot = Resolve-FullPath $OutputRoot
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$stageDir = Join-Path $OutputRoot "stage"
$installerDir = Join-Path $OutputRoot "installer"
$installedDir = Join-Path $OutputRoot "installed"
$verificationRoot = Join-Path $OutputRoot "verification"
$pyInstallerDist = Join-Path $OutputRoot "pyinstaller-dist"
$pyInstallerWork = Join-Path $OutputRoot "pyinstaller-work"
$generatedDir = Join-Path $OutputRoot "generated"
$buildVenv = Join-Path $OutputRoot "build-venv"

Reset-GeneratedDir -Path $stageDir -OutputRoot $OutputRoot
Reset-GeneratedDir -Path $installerDir -OutputRoot $OutputRoot
Reset-GeneratedDir -Path $verificationRoot -OutputRoot $OutputRoot
Reset-GeneratedDir -Path $pyInstallerDist -OutputRoot $OutputRoot
Reset-GeneratedDir -Path $pyInstallerWork -OutputRoot $OutputRoot
Reset-GeneratedDir -Path $generatedDir -OutputRoot $OutputRoot

$basePython = Join-Path $ProjectRoot ".venv-win\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
  throw "Validated Python environment not found: $basePython"
}

$buildPython = Join-Path $buildVenv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
  Write-Host "Creating isolated packaging venv: $buildVenv"
  & $basePython -m venv $buildVenv
}
if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
  throw "Failed to create packaging venv: $buildVenv"
}

Write-Host "[1/7] Installing packaging dependencies"
$requirements = Join-Path $ProjectRoot "requirements.lock.txt"
& $buildPython -m pip install --disable-pip-version-check --timeout 180 --retries 10 -r $requirements pyinstaller==6.11.1
if ($LASTEXITCODE -ne 0) {
  throw "pip install failed. ExitCode=$LASTEXITCODE"
}

Write-Host "[2/7] Building Python engine executable"
$engineEntry = Join-Path $ProjectRoot "appshell\core\python_engine\engine_main.py"
$engineModulePath = Join-Path $ProjectRoot "appshell\core\python_engine"
$pythonBasePrefix = (& $buildPython -c "import sys; print(sys.base_prefix)").Trim()
$condaLibraryBin = Join-Path $pythonBasePrefix "Library\bin"
$condaDllNames = @("ffi.dll", "liblzma.dll", "libbz2.dll", "sqlite3.dll", "tcl86t.dll", "tk86t.dll", "tbb12.dll")
$pyInstallerBinaryArgs = @()
foreach ($dllName in $condaDllNames) {
  $dllPath = Join-Path $condaLibraryBin $dllName
  if (Test-Path -LiteralPath $dllPath -PathType Leaf) {
    $pyInstallerBinaryArgs += @("--add-binary", "$dllPath;.")
  }
}
& $buildPython -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --name anomaly_engine `
  --distpath $pyInstallerDist `
  --workpath $pyInstallerWork `
  --specpath $generatedDir `
  --paths $ProjectRoot `
  --paths $engineModulePath `
  --hidden-import action_catalog `
  --hidden-import engine_core `
  --hidden-import engine_logging `
  --hidden-import engine_protocol `
  --hidden-import engine_service `
  --hidden-import src.training_core `
  --hidden-import src.repair_core `
  --hidden-import src.repair_module `
  --hidden-import pandas `
  --hidden-import numpy `
  --hidden-import lightgbm `
  --hidden-import sklearn `
  --hidden-import joblib `
  --collect-all lightgbm `
  --collect-all scipy `
  --collect-all sklearn `
  --exclude-module matplotlib `
  --exclude-module matplotlib.pyplot `
  --exclude-module matplotlib.backends `
  --exclude-module PIL `
  --exclude-module tkinter `
  --exclude-module streamlit `
  --exclude-module altair `
  --exclude-module shap `
  --exclude-module numba `
  $pyInstallerBinaryArgs `
  $engineEntry
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed. ExitCode=$LASTEXITCODE"
}

$engineDist = Join-Path $pyInstallerDist "anomaly_engine"
$stageEngine = Join-Path $stageDir "python_engine"
if (-not (Test-Path -LiteralPath (Join-Path $engineDist "anomaly_engine.exe") -PathType Leaf)) {
  throw "PyInstaller output is missing anomaly_engine.exe: $engineDist"
}
Copy-Item -LiteralPath $engineDist -Destination $stageEngine -Recurse -Force

Write-Host "[3/7] Building Wails desktop executable"
$backendDir = Join-Path $ProjectRoot "appshell\backend"
$mainExe = Join-Path $stageDir "AnomalyDetectionRepair.exe"
$env:PATH = (Join-Path $ProjectRoot ".venv-win\Scripts") + ";" + $env:PATH
Push-Location $backendDir
try {
  & go build -trimpath -ldflags "-s -w" -o $mainExe ./cmd/wails
  if ($LASTEXITCODE -ne 0) {
    throw "go build failed. ExitCode=$LASTEXITCODE"
  }
} finally {
  Pop-Location
}

Write-Host "[4/7] Copying frontend, samples, and docs"
$stageFrontend = Join-Path $stageDir "frontend"
New-Item -ItemType Directory -Force -Path $stageFrontend | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "appshell\frontend\*") -Destination $stageFrontend -Recurse -Force

$stageSamples = Join-Path $stageDir "samples"
New-Item -ItemType Directory -Force -Path $stageSamples | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "data\experiments\m1_stroke\corrupted.csv") -Destination (Join-Path $stageSamples "m1_stroke_corrupted.csv") -Force

$stageDocs = Join-Path $stageDir "docs"
New-Item -ItemType Directory -Force -Path $stageDocs | Out-Null
foreach ($doc in @("README.md", "ENVIRONMENT.md", "appshell\README.md", "appshell\backend\README.md")) {
  $source = Join-Path $ProjectRoot $doc
  if (Test-Path -LiteralPath $source -PathType Leaf) {
    $safeName = ($doc -replace '[\\/]', '_')
    Copy-Item -LiteralPath $source -Destination (Join-Path $stageDocs $safeName) -Force
  }
}

$packageReadme = @"
# Anomaly Detection Repair Windows Package

Run `AnomalyDetectionRepair.exe` to start the desktop app.

Included:
- `python_engine/anomaly_engine.exe`: packaged Python engine.
- `frontend/`: Wails frontend assets.
- `samples/m1_stroke_corrupted.csv`: small sample CSV for smoke verification.

Runtime data is written under `%LOCALAPPDATA%\AnomalyDetectionRepair` unless overridden with `APPSHELL_DATA_ROOT`.
"@
Set-Content -LiteralPath (Join-Path $stageDir "PACKAGE_README.md") -Value $packageReadme -Encoding UTF8

Write-Host "[5/7] Running package smoke test on staged directory"
$smokeScript = Join-Path $PSScriptRoot "smoke_test_package.ps1"
Invoke-SmokeTestScript -SmokeScript $smokeScript -PackageDir $stageDir -OutputRoot (Join-Path $verificationRoot "stage") -SkipGuiLaunch:$SkipGuiLaunch

$installerExe = ""
if (-not $SkipInstaller) {
  Write-Host "[6/7] Building Inno Setup installer"
  $iscc = Ensure-InnoCompiler
  $template = Get-Content -LiteralPath (Join-Path $PSScriptRoot "installer.iss.in") -Raw -Encoding UTF8
  $generatedIss = Join-Path $generatedDir "installer.generated.iss"
  $template = $template.Replace("{{VERSION}}", $Version)
  $template = $template.Replace("{{STAGE_DIR}}", $stageDir)
  $template = $template.Replace("{{INSTALLER_OUTPUT_DIR}}", $installerDir)
  Set-Content -LiteralPath $generatedIss -Value $template -Encoding UTF8

  & $iscc $generatedIss
  if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compiler failed. ExitCode=$LASTEXITCODE"
  }

  $installerExe = Join-Path $installerDir "AnomalyDetectionRepairSetup-$Version.exe"
  if (-not (Test-Path -LiteralPath $installerExe -PathType Leaf)) {
    throw "Installer output was not found: $installerExe"
  }

  Write-Host "[7/7] Installing package silently and smoking installed directory"
  Reset-GeneratedDir -Path $installedDir -OutputRoot $OutputRoot
  $installLog = Join-Path $verificationRoot "inno-install.log"
  $installArgs = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS", "/DIR=$installedDir", "/LOG=$installLog")
  & $installerExe @installArgs
  $installExitCode = $LASTEXITCODE
  Wait-InstalledPackage -InstalledDir $installedDir -InstallLog $installLog
  if ($installExitCode -ne 0) {
    $installedMainExe = Join-Path $installedDir "AnomalyDetectionRepair.exe"
    $installLogText = ""
    if (Test-Path -LiteralPath $installLog -PathType Leaf) {
      $installLogText = Get-Content -LiteralPath $installLog -Raw -Encoding UTF8
    }
    if ((-not (Test-Path -LiteralPath $installedMainExe -PathType Leaf)) -or ($installLogText -notmatch "Installation process succeeded")) {
      throw "Silent installer run failed. ExitCode=$installExitCode"
    }
    Write-Warning "Silent installer returned ExitCode=$installExitCode, but installation log and installed files indicate success."
  }
  Invoke-SmokeTestScript -SmokeScript $smokeScript -PackageDir $installedDir -OutputRoot (Join-Path $verificationRoot "installed") -SkipGuiLaunch:$SkipGuiLaunch
} else {
  Write-Host "[6/7] Installer build skipped"
  Write-Host "[7/7] Installed smoke test skipped"
}

$manifest = [ordered]@{
  version = $Version
  output_root = $OutputRoot
  stage_dir = $stageDir
  main_exe = $mainExe
  engine_exe = Join-Path $stageEngine "anomaly_engine.exe"
  installer_exe = $installerExe
  installed_dir = $(if ($SkipInstaller) { "" } else { $installedDir })
  verification_root = $verificationRoot
  skip_installer = [bool]$SkipInstaller
  skip_gui_launch = [bool]$SkipGuiLaunch
  built_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
}
$manifestPath = Join-Path $OutputRoot "build_manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Windows package build completed."
Write-Host "Manifest: $manifestPath"
if ($installerExe) {
  Write-Host "Installer: $installerExe"
}
