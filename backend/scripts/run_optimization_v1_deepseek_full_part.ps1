param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4)]
    [int]$Part
)

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
$PlanPath = Join-Path $OutputRoot "OPT-V1-DEEPSEEKV4-FULL-001_partitions.json"
$LogRoot = Join-Path $RunRoot "logs"

$Plan = Get-Content -Raw -LiteralPath $PlanPath | ConvertFrom-Json
$PartPlan = @($Plan.parts | Where-Object { [int]$_.part -eq $Part })
if ($PartPlan.Count -ne 1) {
    throw "Partition plan does not contain exactly one entry for part $Part"
}
$PartPlan = $PartPlan[0]
$RunId = [string]$PartPlan.run_id
$RunDir = Join-Path $OutputRoot $RunId
$ScoresPath = Join-Path $RunDir "$RunId`_scores.json"
$PartialPath = Join-Path $RunDir "$RunId.partial.json"
$PartLabel = $Part.ToString("00")

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OptimizationDeepSeekFullPartKeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]1
[OptimizationDeepSeekFullPartKeepAwake]::SetThreadExecutionState(
    $EsContinuous -bor $EsSystemRequired
) | Out-Null

try {
    if (Test-Path -LiteralPath $ScoresPath) {
        Write-Output "[$RunId] completed scores already exist; skipping."
        exit 0
    }

    $MaxControllerAttempts = 50
    $ExistingAttemptCount = @(
        Get-ChildItem -LiteralPath $LogRoot -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -like "deepseek_full_part$PartLabel`_controller_attempt*.log"
            }
    ).Count
    for ($Attempt = 1; $Attempt -le $MaxControllerAttempts; $Attempt += 1) {
        $AttemptNumber = $ExistingAttemptCount + $Attempt
        $Arguments = @(
            $ManagePy,
            "run_openai_judge",
            "--provider", [string]$Plan.provider,
            "--input-run", $SourcePath,
            "--output-dir", $OutputRoot,
            "--run-id", $RunId,
            "--mode", "both",
            "--seed", [string]$Plan.seed,
            "--judge-model", [string]$Plan.judge_model,
            "--reasoning-effort", [string]$Plan.reasoning_effort,
            "--max-output-tokens", [string]$Plan.max_output_tokens,
            "--timeout-seconds", [string]$Plan.timeout_seconds,
            "--max-retries", [string]$Plan.max_retries,
            "--base-url", "https://api.deepseek.com",
            "--fail-on-error"
        )
        foreach ($QraId in @($PartPlan.qra_ids)) {
            $Arguments += @("--qra-id", [string]$QraId)
        }
        if (Test-Path -LiteralPath $PartialPath) {
            $Arguments += "--resume"
        }

        $LogPath = Join-Path $LogRoot (
            "deepseek_full_part$PartLabel`_controller_attempt$AttemptNumber.log"
        )
        Write-Output (
            "[$RunId] controller attempt $AttemptNumber started at " +
            [DateTimeOffset]::UtcNow.ToString("o")
        )
        $PreviousErrorActionPreference = $ErrorActionPreference
        try {
            # Windows PowerShell promotes native stderr to ErrorRecord objects.
            # Preserve those records in the audit log without aborting retries.
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
    [OptimizationDeepSeekFullPartKeepAwake]::SetThreadExecutionState(
        $EsContinuous
    ) | Out-Null
}
