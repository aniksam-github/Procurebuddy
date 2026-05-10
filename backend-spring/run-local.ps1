$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$envCandidates = @(
    (Join-Path $scriptDir "..\\backend\\.env"),
    (Join-Path $scriptDir ".env"),
    (Join-Path $scriptDir "..\\.env")
)

$envFiles = $envCandidates | Where-Object { Test-Path $_ }

if (-not $envFiles) {
    Write-Error "No .env file found. Expected one of: $($envCandidates -join ', ')"
}

foreach ($envFile in $envFiles) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }

        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            return
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()

        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }

        Set-Item -Path "Env:$name" -Value $value
    }
}

Set-Location $scriptDir
mvn.cmd spring-boot:run
