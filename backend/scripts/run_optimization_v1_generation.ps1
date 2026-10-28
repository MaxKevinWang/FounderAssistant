param()

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptRoot
$RepoRoot = Split-Path -Parent $BackendRoot
$PythonExe = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$ManagePy = Join-Path $BackendRoot "manage.py"
$QraDir = Join-Path $RepoRoot "docs\evaluation\optimization_dataset\qra\frozen"
$OutputDir = Join-Path $RepoRoot "docs\evaluation\runs\optimization_v1\generation"
$PromptPath = Join-Path $RepoRoot "docs\evaluation\formal_dataset\prompts\formal_answer_generation_prompt_v1.0.json"

$env:LOCAL_LLM_TIMEOUT_SECONDS = "600"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OptimizationGenerationKeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]0x00000001
[OptimizationGenerationKeepAwake]::SetThreadExecutionState(
    $EsContinuous -bor $EsSystemRequired
) | Out-Null

function Invoke-OptimizationRun {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string[]]$Models,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$Temperature,
        [Parameter(Mandatory = $true)][string]$MaxTokens,
        [string]$ReasoningEffort = ""
    )

    $FinalPath = Join-Path $OutputDir "$RunId.json"
    $PartialPath = Join-Path $OutputDir "$RunId.partial.json"
    if (Test-Path -LiteralPath $FinalPath) {
        Write-Output "[$RunId] completed artifact already exists; skipping."
        return
    }

    $Arguments = @(
        $ManagePy,
        "run_formal_development_eval",
        "--qra-dir", $QraDir,
        "--output-dir", $OutputDir,
        "--prompt-path", $PromptPath,
        "--mode", $Mode,
        "--temperature", $Temperature,
        "--top-k", "5",
        "--max-tokens", $MaxTokens,
        "--schedule-seed", "20260818",
        "--run-id", $RunId,
        "--fail-on-error"
    )
    foreach ($Model in $Models) {
        $Arguments += @("--model", $Model)
    }
    if ($ReasoningEffort) {
        $Arguments += @("--reasoning-effort", $ReasoningEffort)
    }
    if (Test-Path -LiteralPath $PartialPath) {
        $Arguments += "--resume"
    }

    Write-Output "[$RunId] starting at $([DateTimeOffset]::UtcNow.ToString('o'))"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Generation run $RunId stopped with exit code $LASTEXITCODE"
    }
    Write-Output "[$RunId] completed at $([DateTimeOffset]::UtcNow.ToString('o'))"
}

try {
    Invoke-OptimizationRun `
        -RunId "OPT-V1-CORE-BOTH-001" `
        -Models @(
            "gemma4:31b",
            "gemma4:26b",
            "gpt-oss:20b",
            "magistral:latest",
            "olmo-3.1:32b-think",
            "qwen3.6:latest"
        ) `
        -Mode "both" `
        -Temperature "0.1" `
        -MaxTokens "900"

    Invoke-OptimizationRun `
        -RunId "OPT-V1-KIMI-RAG-001" `
        -Models @("cloud.moonshot/kimi-k3") `
        -Mode "rag" `
        -Temperature "1.0" `
        -MaxTokens "8192" `
        -ReasoningEffort "low"

    Invoke-OptimizationRun `
        -RunId "OPT-V1-KIMI-BASELINE-001" `
        -Models @("cloud.moonshot/kimi-k3") `
        -Mode "baseline" `
        -Temperature "1.0" `
        -MaxTokens "8192" `
        -ReasoningEffort "low"
}
finally {
    [OptimizationGenerationKeepAwake]::SetThreadExecutionState($EsContinuous) |
        Out-Null
}
