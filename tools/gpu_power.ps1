<#
    gpu_power.ps1 — the elevated half of the "GPU Power Limit" feature.

    Run by the Scheduled Task `SynesthesiaAIVideoDirectorGpuPower` (registered once by
    register_gpu_power_task.bat, /RU SYSTEM /RL HIGHEST).  The unelevated app
    writes gpu_power_plan.json, fires `schtasks /run`, and polls for a
    gpu_power_result.json carrying the plan's nonce.

    SECURITY: the plan file is written by an unelevated process and consumed
    here with SYSTEM rights, so nothing in it is trusted.  Indices and wattages
    are cast to [int] and clamped against limits this script queries from
    nvidia-smi itself; the nvidia-smi command line is built from those
    validated integers only — no file content is ever interpolated into it.
    The worst a hostile plan file can do is set a legal power limit on a local
    GPU.
#>

$ErrorActionPreference = 'Stop'

$ExchangeDir = Join-Path $env:ProgramData 'Synesthesia AI Video Director\GpuPowerExchange'
$PlanPath   = Join-Path $ExchangeDir 'gpu_power_plan.json'
$ResultPath = Join-Path $ExchangeDir 'gpu_power_result.json'
New-Item -ItemType Directory -Path $ExchangeDir -Force | Out-Null


function Write-Result {
    param([hashtable]$Payload)
    $Payload['timestamp'] = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $json = $Payload | ConvertTo-Json -Depth 6
    # UTF8 *without* BOM — json.load() on the Python side chokes on one.
    [System.IO.File]::WriteAllText(
        $ResultPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}


function Get-NvidiaSmi {
    # As SYSTEM the user PATH is not in play, so fall back to the two places
    # the driver actually installs nvidia-smi.exe.
    $cmd = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        (Join-Path $env:SystemRoot  'System32\nvidia-smi.exe'),
        (Join-Path ${env:ProgramFiles} 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    )) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    throw 'nvidia-smi.exe not found.'
}


function Get-GpuTable {
    param([string]$Smi)
    $fields = 'index,name,power.limit,power.default_limit,power.min_limit,power.max_limit'
    $rows   = & $Smi "--query-gpu=$fields" '--format=csv,noheader,nounits'
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi query failed (exit $LASTEXITCODE)." }

    $table = @{}
    foreach ($row in $rows) {
        $p = $row -split '\s*,\s*'
        if ($p.Count -lt 6) { continue }
        $table[[int]$p[0]] = @{
            name    = $p[1]
            limit   = [double]$p[2]
            default = [double]$p[3]
            min     = [double]$p[4]
            max     = [double]$p[5]
        }
    }
    return $table
}


$nonce = ''
try {
    if (-not (Test-Path -LiteralPath $PlanPath)) { throw "No plan file at $PlanPath." }

    $plan  = Get-Content -Raw -LiteralPath $PlanPath | ConvertFrom-Json
    $nonce = [string]$plan.nonce
    $mode  = [string]$plan.mode
    if ($mode -ne 'apply' -and $mode -ne 'restore') { throw "Bad mode '$mode'." }

    $smi  = Get-NvidiaSmi
    $gpus = Get-GpuTable -Smi $smi

    # Build the work list as validated integers.  Restore ignores the plan's
    # targets entirely and puts *every* card back to its own stock limit, so a
    # truncated or stale plan can never leave a GPU capped.
    $work = @()
    if ($mode -eq 'restore') {
        foreach ($idx in ($gpus.Keys | Sort-Object)) {
            $work += @{ index = [int]$idx; watts = [int][math]::Round($gpus[$idx].default) }
        }
    } else {
        foreach ($t in @($plan.targets)) {
            $idx = [int]$t.index
            if (-not $gpus.ContainsKey($idx)) { continue }
            $w   = [int]$t.watts
            $lo  = [int][math]::Ceiling($gpus[$idx].min)
            $hi  = [int][math]::Floor($gpus[$idx].max)
            if ($w -lt $lo) { $w = $lo }
            if ($w -gt $hi) { $w = $hi }
            $work += @{ index = $idx; watts = $w }
        }
    }

    $entries = @()
    $allOk   = $true
    foreach ($item in $work) {
        $i = [int]$item.index      # re-cast: only integers reach the command line
        $w = [int]$item.watts
        $out = (& $smi '-i' "$i" '-pl' "$w" 2>&1) -join ' '
        $ok  = ($LASTEXITCODE -eq 0)
        if (-not $ok) { $allOk = $false }
        $entries += [ordered]@{
            index   = $i
            name    = $gpus[$i].name
            watts   = $w
            ok      = $ok
            message = $out.Trim()
        }
    }

    # Re-query so the app reports what the hardware actually holds, not what we
    # asked for.
    $after = Get-GpuTable -Smi $smi
    foreach ($e in $entries) {
        $e['applied'] = [int][math]::Round($after[[int]$e.index].limit)
    }

    Write-Result @{
        nonce   = $nonce
        mode    = $mode
        ok      = $allOk
        entries = $entries
    }
    if (-not $allOk) { exit 1 }
    exit 0
}
catch {
    Write-Result @{
        nonce   = $nonce
        mode    = 'error'
        ok      = $false
        error   = $_.Exception.Message
        entries = @()
    }
    exit 1
}
