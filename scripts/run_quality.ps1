$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing local Python environment: $python"
}

function Invoke-QualityCommand {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Quality command failed: python $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    Invoke-QualityCommand -Arguments @('-m', 'pytest')
    Invoke-QualityCommand -Arguments @('-m', 'ruff', 'check', '.')
    Invoke-QualityCommand -Arguments @('-m', 'ruff', 'format', '--check', '.')
    Invoke-QualityCommand -Arguments @('-m', 'mypy')
    Invoke-QualityCommand -Arguments @('-m', 'build')
}
finally {
    Pop-Location
}

Write-Output 'All local GAZELINK quality gates passed.'
