Param(
  [Parameter(Mandatory = $true)]
  [string]$PackageDir,
  [string]$OutputRoot = "",
  [switch]$SkipGuiLaunch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-FullPath {
  Param([Parameter(Mandatory = $true)][string]$Path)
  return [System.IO.Path]::GetFullPath($Path)
}

function Invoke-EngineRequest {
  Param(
    [Parameter(Mandatory = $true)][string]$EngineExe,
    [Parameter(Mandatory = $true)][hashtable]$Request,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$TimeoutSeconds = 180
  )

  $json = $Request | ConvertTo-Json -Depth 30 -Compress
  $stdoutPath = Join-Path $OutputDir "$Name.stdout.json"
  $stderrPath = Join-Path $OutputDir "$Name.stderr.log"

  $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $EngineExe
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardInput = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true

  $process = [System.Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  if (-not $process.Start()) {
    throw "Failed to start engine executable: $EngineExe"
  }

  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $process.StandardInput.Write($json)
  $process.StandardInput.Close()

  $timeoutMs = [Math]::Max(1, $TimeoutSeconds) * 1000
  $timedOut = -not $process.WaitForExit($timeoutMs)
  if ($timedOut) {
    try {
      $process.Kill($true)
    } catch {
      try {
        $process.Kill()
      } catch {
        Write-Warning "Failed to kill timed-out engine process $($process.Id): $($_.Exception.Message)"
      }
    }
    $process.WaitForExit(10000) | Out-Null
  } else {
    $process.WaitForExit()
  }

  $stdout = $stdoutTask.GetAwaiter().GetResult()
  $stderr = $stderrTask.GetAwaiter().GetResult()

  Set-Content -LiteralPath $stdoutPath -Value $stdout -Encoding UTF8
  Set-Content -LiteralPath $stderrPath -Value $stderr -Encoding UTF8

  if ($timedOut) {
    throw "Engine request '$Name' timed out after $TimeoutSeconds seconds. See $stdoutPath and $stderrPath"
  }

  if ($process.ExitCode -ne 0) {
    throw "Engine request '$Name' exited with code $($process.ExitCode). See $stderrPath"
  }

  $body = ($stdout -split "`r?`n" | Where-Object { $_.Trim() }) | Select-Object -First 1
  if (-not $body) {
    throw "Engine request '$Name' returned empty stdout."
  }
  return $body | ConvertFrom-Json
}

function Assert-EngineOk {
  Param(
    [Parameter(Mandatory = $true)]$Response,
    [Parameter(Mandatory = $true)][string]$Name
  )
  if ($Response.status -ne "ok") {
    $payload = $Response | ConvertTo-Json -Depth 20
    throw "Engine request '$Name' failed: $payload"
  }
}

$PackageDir = Resolve-FullPath $PackageDir
if (-not $OutputRoot) {
  $OutputRoot = Join-Path $PackageDir "verification"
}
$OutputRoot = Resolve-FullPath $OutputRoot
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$mainExe = Join-Path $PackageDir "AnomalyDetectionRepair.exe"
$engineExe = Join-Path $PackageDir "python_engine\anomaly_engine.exe"
$sampleCsv = Join-Path $PackageDir "samples\m1_stroke_corrupted.csv"

foreach ($requiredPath in @($mainExe, $engineExe, $sampleCsv)) {
  if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
    throw "Required package file is missing: $requiredPath"
  }
}

$health = Invoke-EngineRequest -EngineExe $engineExe -OutputDir $OutputRoot -Name "health" -Request @{
  task_id = "pkg-health"
  action = "health"
  payload = @{}
}
Assert-EngineOk -Response $health -Name "health"

$scan = Invoke-EngineRequest -EngineExe $engineExe -OutputDir $OutputRoot -Name "scan_file" -Request @{
  task_id = "pkg-scan"
  action = "scan_file"
  payload = @{
    csv_path = $sampleCsv
    max_issues = 50
    preview_limit = 5
  }
}
Assert-EngineOk -Response $scan -Name "scan_file"

$issueCount = [int]$scan.result.issue_count
if ($issueCount -le 0) {
  throw "scan_file did not find any issues in the packaged sample CSV."
}

$issues = @($scan.result.issues)
$safeIssue = $issues |
  Where-Object { $_.repair_supported -eq $true -and ($_.issue_type -eq "missing_values" -or $_.issue_type -eq "rare_category") } |
  Select-Object -First 1
if (-not $safeIssue) {
  $safeIssue = $issues | Where-Object { $_.repair_supported -eq $true } | Select-Object -First 1
}
if (-not $safeIssue) {
  throw "scan_file returned issues, but none were repair-supported."
}

$repairOutputDir = Join-Path $OutputRoot "repair-output"
$rollbackDir = Join-Path $repairOutputDir ".rollback"
New-Item -ItemType Directory -Force -Path $repairOutputDir | Out-Null

$repair = Invoke-EngineRequest -EngineExe $engineExe -OutputDir $OutputRoot -Name "repair_batch" -Request @{
  task_id = "pkg-repair"
  action = "repair_batch"
  payload = @{
    csv_path = $sampleCsv
    issue_ids = @($safeIssue.issue_id)
    write_output = $true
    output_dir = $repairOutputDir
    enable_rollback = $true
    rollback_dir = $rollbackDir
  }
}
Assert-EngineOk -Response $repair -Name "repair_batch"

$repairedCsv = [string]$repair.result.output_csv
$manifestPath = [string]$repair.result.rollback.manifest_path
if (-not (Test-Path -LiteralPath $repairedCsv -PathType Leaf)) {
  throw "repair_batch did not generate repaired CSV: $repairedCsv"
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
  throw "repair_batch did not generate rollback manifest: $manifestPath"
}

$guiStatus = "skipped"
$guiPid = ""
if (-not $SkipGuiLaunch) {
  $process = Start-Process -FilePath $mainExe -WorkingDirectory $PackageDir -PassThru
  Start-Sleep -Seconds 8
  $guiPid = [string]$process.Id
  if ($process.HasExited) {
    throw "GUI process exited during launch smoke test. ExitCode=$($process.ExitCode)"
  }
  $guiStatus = "started"
  Stop-Process -Id $process.Id -Force
}

$reportPath = Join-Path $OutputRoot "verification.md"
$report = @"
# Windows Package Smoke Verification

- Package dir: $PackageDir
- Main exe: $mainExe
- Engine exe: $engineExe
- Sample CSV: $sampleCsv
- Health status: $($health.status)
- Scan issue count: $issueCount
- Repaired issue id: $($safeIssue.issue_id)
- Repaired CSV: $repairedCsv
- Rollback manifest: $manifestPath
- GUI launch status: $guiStatus
- GUI process id: $guiPid

Notes:
- The automated smoke test validates the packaged engine health, CSV scan, repair execution, output CSV, rollback manifest, and main GUI process launch.
- File-dialog clicking and visual result inspection remain an operator check in the running desktop UI.
"@
Set-Content -LiteralPath $reportPath -Value $report -Encoding UTF8

$summary = [ordered]@{
  package_dir = $PackageDir
  main_exe = $mainExe
  engine_exe = $engineExe
  sample_csv = $sampleCsv
  health_status = $health.status
  scan_issue_count = $issueCount
  repaired_issue_id = [string]$safeIssue.issue_id
  repaired_csv = $repairedCsv
  rollback_manifest = $manifestPath
  gui_launch_status = $guiStatus
  gui_pid = $guiPid
  report = $reportPath
}
$summaryPath = Join-Path $OutputRoot "verification_summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host "Smoke verification completed: $reportPath"
