param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptRoot
$RepoRoot = Split-Path -Parent $BackendRoot
$PythonExe = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$ManagePy = Join-Path $BackendRoot "manage.py"
$RunRoot = Join-Path $RepoRoot "docs\evaluation\runs\optimization_v1"
$SourcePath = Join-Path $RunRoot "generation\OPT-V1-ALL-AUDITED-001.json"
$OutputRoot = Join-Path $RunRoot "judge\deepseek_api"
$LogRoot = Join-Path $RunRoot "logs"
$RunId = "OPT-V1-DEEPSEEKV4-CALIBRATION12-001"
$RunDir = Join-Path $OutputRoot $RunId
$ScoresPath = Join-Path $RunDir "$RunId`_scores.json"
$PartialPath = Join-Path $RunDir "$RunId.partial.json"

$QraIds = @(
    "OPTLFS-F-001",
    "OPTLFS-S-002",
    "OPTLFS-MHR-003",
    "OPTLFS-II-002",
    "OPTFP-NC-002",
    "OPTFP-TS-001",
    "OPTFP-U-002",
    "OPTFP-F-004",
    "OPTMV-S-003",
    "OPTMV-MHR-001",
    "OPTMV-II-003",
    "OPTMV-NC-001"
)

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OptimizationDeepSeekCalibrationKeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]1
[OptimizationDeepSeekCalibrationKeepAwake]::SetThreadExecutionState(
    $EsContinuous -bor $EsSystemRequired
) | Out-Null

try {
    if (Test-Path -LiteralPath $ScoresPath) {
        Write-Output "[$RunId] completed scores already exist; skipping."
        exit 0
    }

    $MaxControllerAttempts = 25
    $ExistingAttemptCount = @(
        Get-ChildItem -LiteralPath $LogRoot -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -like "deepseek_calibration12_controller_attempt*.log"
            }
    ).Count
    for ($Attempt = 1; $Attempt -le $MaxControllerAttempts; $Attempt += 1) {
        $AttemptNumber = $ExistingAttemptCount + $Attempt
        $Arguments = @(
            $ManagePy,
            "run_openai_judge",
            "--provider", "deepseek",
            "--input-run", $SourcePath,
            "--output-dir", $OutputRoot,
            "--run-id", $RunId,
            "--mode", "both",
            "--seed", "20260825",
            "--fail-on-error"
        )
        foreach ($QraId in $QraIds) {
            $Arguments += @("--qra-id", $QraId)
        }
        if (Test-Path -LiteralPath $PartialPath) {
            $Arguments += "--resume"
        }

        $LogPath = Join-Path $LogRoot (
            "deepseek_calibration12_controller_attempt$AttemptNumber.log"
        )
        Write-Output (
            "[$RunId] controller attempt $AttemptNumber started at " +
            [DateTimeOffset]::UtcNow.ToString("o")
        )
        $PreviousErrorActionPreference = $ErrorActionPreference
        try {
            # Windows PowerShell promotes native stderr to ErrorRecord objects.
            # Keep those records in the audit log without aborting this retry loop.
            $ErrorActionPreference = "Continue"
            & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $LogPath
            $ExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }

        if ($ExitCode -eq 0 -and (Test-Path -LiteralPath $ScoresPath)) {
            Write-Output (
                "[$RunId] completed at " +
                [DateTimeOffset]::UtcNow.ToString("o")
            )
            exit 0
        }

        Write-Output (
            "[$RunId] attempt $AttemptNumber stopped with exit code $ExitCode; " +
            "the next attempt will resume its checkpoint."
        )
        Start-Sleep -Seconds 3
    }

    throw "$RunId did not complete after $MaxControllerAttempts controller attempts"
}
finally {
    [OptimizationDeepSeekCalibrationKeepAwake]::SetThreadExecutionState(
        $EsContinuous
    ) | Out-Null
}
