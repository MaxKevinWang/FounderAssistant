param()

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptRoot
$RepoRoot = Split-Path -Parent $BackendRoot
$PythonExe = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$WorkerPath = Join-Path $ScriptRoot "run_optimization_v1_deepseek_full_part.ps1"
$MergePath = Join-Path $ScriptRoot "merge_optimization_v1_deepseek_full.py"
$RunRoot = Join-Path $RepoRoot "docs\evaluation\runs\optimization_v1"
$LogRoot = Join-Path $RunRoot "logs"
$OutputRoot = Join-Path $RunRoot "judge\deepseek_api"
$SupervisorLog = Join-Path $LogRoot "deepseek_full_supervisor.log"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OptimizationDeepSeekFullKeepAwake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
$EsContinuous = [uint32]2147483648
$EsSystemRequired = [uint32]1
[OptimizationDeepSeekFullKeepAwake]::SetThreadExecutionState(
    $EsContinuous -bor $EsSystemRequired
) | Out-Null

try {
    & $PythonExe $MergePath --validate-plan 2>&1 |
        Tee-Object -FilePath $SupervisorLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek full-run partition validation failed"
    }

    $Processes = @()
    foreach ($Part in 1..4) {
        $PartLabel = $Part.ToString("00")
        $RunId = "OPT-V1-DEEPSEEKV4-FULL-PART$PartLabel-001"
        $ScoresPath = Join-Path (Join-Path $OutputRoot $RunId) "$RunId`_scores.json"
        if (Test-Path -LiteralPath $ScoresPath) {
            "[$RunId] final scores already exist; worker not started." |
                Tee-Object -FilePath $SupervisorLog -Append
            continue
        }

        $StdoutPath = Join-Path $LogRoot "deepseek_full_part$PartLabel`_worker.stdout.log"
        $StderrPath = Join-Path $LogRoot "deepseek_full_part$PartLabel`_worker.stderr.log"
        $Process = Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList @(
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", $WorkerPath,
                "-Part", [string]$Part
            ) `
            -WindowStyle Hidden `
            -RedirectStandardOutput $StdoutPath `
            -RedirectStandardError $StderrPath `
            -PassThru
        $Processes += [PSCustomObject]@{
            Part = $Part
            RunId = $RunId
            Process = $Process
        }
        "[$RunId] worker process $($Process.Id) started." |
            Tee-Object -FilePath $SupervisorLog -Append
    }

    foreach ($Record in $Processes) {
        $Record.Process.WaitForExit()
        $Record.Process.Refresh()
        $ExitCode = $Record.Process.ExitCode
        "[$($Record.RunId)] worker exited with code $ExitCode." |
            Tee-Object -FilePath $SupervisorLog -Append
        $ScoresPath = Join-Path (
            Join-Path $OutputRoot $Record.RunId
        ) "$($Record.RunId)_scores.json"
        if (Test-Path -LiteralPath $ScoresPath) {
            "[$($Record.RunId)] completed scores verified." |
                Tee-Object -FilePath $SupervisorLog -Append
            continue
        }
        if ($null -eq $ExitCode -or $ExitCode -ne 0) {
            throw "$($Record.RunId) worker failed with exit code $ExitCode"
        }
        throw "$($Record.RunId) exited without a completed scores artifact"
    }

    & $PythonExe $MergePath 2>&1 |
        Tee-Object -FilePath $SupervisorLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek full-run merge failed"
    }
    "DeepSeek full evaluation and merge completed." |
        Tee-Object -FilePath $SupervisorLog -Append
}
finally {
    [OptimizationDeepSeekFullKeepAwake]::SetThreadExecutionState(
        $EsContinuous
    ) | Out-Null
}
