param()

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptRoot
$RepoRoot = Split-Path -Parent $BackendRoot
$PythonExe = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$ManagePy = Join-Path $BackendRoot "manage.py"
$RunRoot = Join-Path $RepoRoot "docs\evaluation\runs\optimization_v1"
$OutputRoot = Join-Path $RunRoot "judge\codex"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OptimizationJudgeKeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]1
[OptimizationJudgeKeepAwake]::SetThreadExecutionState(
    $EsContinuous -bor $EsSystemRequired
) | Out-Null

function Invoke-CodexJudge {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Condition,
        [Parameter(Mandatory = $true)][string]$SourceName,
        [Parameter(Mandatory = $true)][string]$Seed
    )

    $RunDir = Join-Path $OutputRoot $RunId
    $ScoresPath = Join-Path $RunDir "$RunId`_scores.json"
    if (Test-Path -LiteralPath $ScoresPath) {
        Write-Output "[$RunId] completed scores already exist; skipping."
        return
    }

    $SourcePath = Join-Path $RunRoot "generation\$SourceName"
    $Arguments = @(
        $ManagePy,
        "run_formal_codex_judge",
        "--source-run", $SourcePath,
        "--output-root", $OutputRoot,
        "--run-id", $RunId,
        "--condition", $Condition,
        "--phase", "first",
        "--first-phase-model-count", "7",
        "--batch-size", "10",
        "--judge-model", "gpt-5.6-sol",
        "--reasoning-effort", "medium",
        "--seed", $Seed,
        "--timeout-seconds", "900",
        "--max-batch-retries", "2",
        "--resume"
    )
    Write-Output "[$RunId] starting at $([DateTimeOffset]::UtcNow.ToString('o'))"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Codex Judge run $RunId stopped with exit code $LASTEXITCODE"
    }
    Write-Output "[$RunId] completed at $([DateTimeOffset]::UtcNow.ToString('o'))"
}

try {
    Invoke-CodexJudge `
        -RunId "OPT-V1-CODEX56-BASELINE-JUDGE-001" `
        -Condition "baseline" `
        -SourceName "OPT-V1-BASELINE-AUDITED-001.json" `
        -Seed "20260820"

    Invoke-CodexJudge `
        -RunId "OPT-V1-CODEX56-RAG-JUDGE-001" `
        -Condition "rag" `
        -SourceName "OPT-V1-RAG-AUDITED-001.json" `
        -Seed "20260821"
}
finally {
    [OptimizationJudgeKeepAwake]::SetThreadExecutionState($EsContinuous) |
        Out-Null
}
