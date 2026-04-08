$ErrorActionPreference = 'Stop'

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
