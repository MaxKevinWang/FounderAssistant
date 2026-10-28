param()

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptRoot
$RepoRoot = Split-Path -Parent $BackendRoot
$PythonExe = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$ManagePy = Join-Path $BackendRoot "manage.py"
$RunRoot = Join-Path $RepoRoot "docs\evaluation\runs\optimization_v1"
$CodexRoot = Join-Path $RunRoot "judge\codex"
$OutputRoot = Join-Path $RunRoot "judge\openai_api"
$SourcePath = Join-Path $RunRoot "generation\OPT-V1-ALL-AUDITED-001.json"

$CodexScorePaths = @(
    (Join-Path $CodexRoot "OPT-V1-CODEX56-BASELINE-JUDGE-001\OPT-V1-CODEX56-BASELINE-JUDGE-001_scores.json"),
    (Join-Path $CodexRoot "OPT-V1-CODEX56-RAG-JUDGE-001\OPT-V1-CODEX56-RAG-JUDGE-001_scores.json")
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OptimizationApiJudgeKeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]1
[OptimizationApiJudgeKeepAwake]::SetThreadExecutionState(
    $EsContinuous -bor $EsSystemRequired
) | Out-Null

function Invoke-Gpt53Judge {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$Seed,
        [string]$TestedModel = "",
        [string]$QraId = ""
    )

    $RunDir = Join-Path $OutputRoot $RunId
    $ScoresPath = Join-Path $RunDir "$RunId`_scores.json"
    $PartialPath = Join-Path $RunDir "$RunId.partial.json"
    if (Test-Path -LiteralPath $ScoresPath) {
        Write-Output "[$RunId] completed scores already exist; skipping."
        return
    }

    $Arguments = @(
        $ManagePy,
        "run_openai_judge",
        "--input-run", $SourcePath,
        "--output-dir", $OutputRoot,
        "--run-id", $RunId,
        "--mode", $Mode,
        "--judge-model", "gpt-5.3-chat-latest",
        "--reasoning-effort", "none",
        "--max-output-tokens", "2500",
        "--timeout-seconds", "360",
        "--max-retries", "3",
        "--seed", $Seed,
        "--fail-on-error"
    )
    if ($TestedModel) {
        $Arguments += @("--tested-model", $TestedModel)
    }
    if ($QraId) {
        $Arguments += @("--qra-id", $QraId)
    }
    if (Test-Path -LiteralPath $PartialPath) {
        $Arguments += "--resume"
    }

    Write-Output "[$RunId] starting at $([DateTimeOffset]::UtcNow.ToString('o'))"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "GPT-5.3 Judge run $RunId stopped with exit code $LASTEXITCODE"
    }
    Write-Output "[$RunId] completed at $([DateTimeOffset]::UtcNow.ToString('o'))"
}

function Test-CodexJudgeComplete {
    param([Parameter(Mandatory = $true)][string]$ScoresPath)

    if (-not (Test-Path -LiteralPath $ScoresPath)) {
        return $false
    }
    try {
        $Payload = Get-Content -LiteralPath $ScoresPath -Raw | ConvertFrom-Json
        return $Payload.status -eq "completed"
    }
    catch {
        return $false
    }
}

function Get-CompletedCodexJudgeCount {
    $Count = 0
    foreach ($ScoresPath in $CodexScorePaths) {
        if (Test-CodexJudgeComplete -ScoresPath $ScoresPath) {
            $Count += 1
        }
    }
    return $Count
}

try {
    $PollCount = 0
    $ReadyCount = Get-CompletedCodexJudgeCount
    while ($ReadyCount -lt 2) {
        if ($PollCount % 10 -eq 0) {
            Write-Output "Waiting for Codex Judge: $ReadyCount/2 score files ready."
        }
        $PollCount += 1
        Start-Sleep -Seconds 30
        $ReadyCount = Get-CompletedCodexJudgeCount
    }

    Write-Output "Both Codex Judge runs completed; starting GPT-5.3 API validation."
    Invoke-Gpt53Judge `
        -RunId "OPT-V1-GPT53-SMOKE-001" `
        -Mode "baseline" `
        -Seed "20260822" `
        -TestedModel "gemma4:31b" `
        -QraId "OPTLFS-F-001"

    Invoke-Gpt53Judge `
        -RunId "OPT-V1-GPT53-JUDGE-001" `
        -Mode "both" `
        -Seed "20260823"
}
finally {
    [OptimizationApiJudgeKeepAwake]::SetThreadExecutionState($EsContinuous) |
        Out-Null
}
