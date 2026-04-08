param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($PSScriptRoot) {
    $repoRoot = $PSScriptRoot
}
else {
    $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

Set-Location $repoRoot

$venvDir = Join-Path $repoRoot 'venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'
$requirementsFile = Join-Path $repoRoot 'requirements.txt'

if (-not (Test-Path $venvPython)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3 -m venv $venvDir
    }
    else {
        $pythonLauncher = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonLauncher) {
            throw 'Python 3 was not found on PATH. Install Python 3.10+ or make sure the Python launcher is available.'
        }

        & python -m venv $venvDir
    }
}

& $venvPython -m pip install --upgrade pip

if (-not $SkipInstall) {
    & $venvPython -m pip install -r $requirementsFile
}

if (Test-Path $activateScript) {
    . $activateScript
}

Write-Host ''
Write-Host 'Virtual environment is ready.'
Write-Host 'Activate it in this shell with:'
Write-Host '  . .\venv\Scripts\Activate.ps1'
Write-Host 'Then train or predict with:'
Write-Host '  python main.py'$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot 'venv'
$PythonExe = Join-Path $VenvPath 'Scripts\python.exe'

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @('py', '-3')
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @('python')
    }

    throw 'Python was not found on PATH. Install Python 3, then run this script again.'
}

if (-not (Test-Path $PythonExe)) {
    Write-Host 'Creating virtual environment...'
    $PythonCommand = Get-PythonCommand
    & $PythonCommand[0] $PythonCommand[1..($PythonCommand.Length - 1)] -m venv $VenvPath
}

Write-Host 'Upgrading pip...'
& $PythonExe -m pip install --upgrade pip

Write-Host 'Installing dependencies...'
& $PythonExe -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')

Write-Host ''
Write-Host 'Setup complete.'
Write-Host "To run the project, use: $PythonExe .\main.py"
