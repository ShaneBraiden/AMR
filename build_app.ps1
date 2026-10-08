<#
.SYNOPSIS
Builds the Windows installer for Antibiotic Resistance Forecast, or for its Studio app.

.DESCRIPTION
  1. installs the build tools into .venv (requirements-build.txt)
  2. trains the seed cache (build\seed_cache): the Forecast app opens it on first launch; the Studio, which
     starts empty, only checks its self-test against it
  3. saves the summary report's fonts (assets\fonts\fonts.css)
  4. packages the app with PyInstaller (dist\<app name>\)
  5. runs the packaged app's self-test: trains every model (10; the Studio 12), compares with the seed cache, writes
     every export. The Studio first reads the bundled workbook as an upload and trains on the data set it confirms.
  6. compiles the installer with Inno Setup (dist\<OutputName>-<version>-Setup.exe)

.EXAMPLE
  .\build_app.ps1                  # the Forecast app: everything
  .\build_app.ps1 -App Studio      # the Studio app: everything
  .\build_app.ps1 -SkipInstaller   # stop after the self-test
#>
param(
    [ValidateSet("Forecast", "Studio")] [string]$App = "Forecast",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Version = "1.0.0"
$Python = ".venv\Scripts\python.exe"
$Workbook = Join-Path $PSScriptRoot "antibiotic trend (2022-2025).xlsx"
$SeedCache = Join-Path $PSScriptRoot "build\seed_cache"

$Apps = @{
    Forecast = @{ Name = "Antibiotic Resistance Forecast"; Profile = "forecast"; Guid = "7C1F3F2E-5B7A-4C1E-9A43-2E6B1F0D9A55";
                  Icon = "app.ico"; Output = "AntibioticResistanceForecast"; SelfTest = "--self-test" }
    Studio   = @{ Name = "Antibiotic Resistance Forecast Studio"; Profile = "entry"; Guid = "E04C8B64-4ED7-4B2B-98F5-43D105BEF67D";
                  Icon = "entry.ico"; Output = "AntibioticResistanceForecastStudio";
                  SelfTest = "--self-test --workbook `"$Workbook`" --seed-cache `"$SeedCache`"" }
}
$Settings = $Apps[$App]
$AppName = $Settings.Name

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

# Run a program and fail on its exit code. Windows PowerShell treats anything a program writes to stderr
# (PyInstaller logs there) as an error, so pass its output through as plain text instead.
function Invoke-Native([string]$What, [string]$Program, [string[]]$Arguments) {
    $ErrorActionPreference = "Continue"
    & $Program @Arguments 2>&1 | ForEach-Object { Write-Host "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

if (-not (Test-Path $Python)) {
    throw "No .venv found. Create it with Python 3.11:  py -3.11 -m venv .venv"
}
Write-Host "Building $AppName $Version" -ForegroundColor Green

Step "Installing the build tools"
Invoke-Native "pip install" $Python @("-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", "requirements-build.txt")

Step "Training the seed cache (about a minute)"
if (Test-Path build\seed_cache) { Remove-Item -Recurse -Force build\seed_cache }
Invoke-Native "Training the seed cache" $Python @("desktop.py", "--build-cache", "build\seed_cache")

Step "Saving the report fonts"
Invoke-Native "Saving the fonts" $Python @("packaging\fetch_fonts.py")

Step "Packaging with PyInstaller (a few minutes)"
Invoke-Native "PyInstaller" $Python @("-m", "PyInstaller", "packaging\app.spec", "--noconfirm", "--clean",
    "--log-level", "WARN", "--distpath", "dist", "--workpath", "build\pyinstaller-$($Settings.Profile)",
    "--", "--profile", $Settings.Profile)
$AppDir = "dist\$AppName"
$Exe = "$AppDir\$AppName.exe"

Step "Self-test of the packaged app (about a minute)"
$TestData = Join-Path $env:TEMP "arf-selftest-$(Get-Random)"
$env:ARF_DATA_DIR = $TestData
try {
    $run = Start-Process -FilePath $Exe -ArgumentList $Settings.SelfTest -Wait -PassThru
} finally {
    Remove-Item Env:ARF_DATA_DIR
}
Get-Content "$TestData\logs\app.log" | Write-Host
if ($run.ExitCode -ne 0) { throw "The packaged app failed its self-test (log above)." }
Remove-Item -Recurse -Force $TestData

$size = (Get-ChildItem $AppDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Packaged app: {0:N0} MB in {1}" -f $size, (Resolve-Path $AppDir))
if ($SkipInstaller) { return }

Step "Compiling the installer (a few minutes)"
$Iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) { throw "Inno Setup 6 is needed to compile the installer. Install it with:  winget install JRSoftware.InnoSetup" }
Invoke-Native "Inno Setup" $Iscc @("/Qp", "/DAppVersion=$Version", "/DAppName=$AppName", "/DAppGuid=$($Settings.Guid)",
    "/DIconFile=$($Settings.Icon)", "/DOutputName=$($Settings.Output)", "packaging\installer.iss")
$Setup = Get-Item "dist\$($Settings.Output)-$Version-Setup.exe"
Write-Host ("Installer: {0} ({1:N0} MB)" -f $Setup.FullName, ($Setup.Length / 1MB)) -ForegroundColor Green
