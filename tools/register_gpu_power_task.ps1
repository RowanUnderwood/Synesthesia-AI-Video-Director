<#
    register_gpu_power_task.ps1 — one-time setup for the GPU power-limit helper.

    Registers the on-demand Scheduled Task `SynesthesiaAIVideoDirectorGpuPower`, which runs
    gpu_power.ps1 as SYSTEM.  That is what lets the unelevated app change GPU
    power limits (an Administrator-only operation) with `schtasks /run` and no
    UAC prompt.

    Needs Administrator, and self-elevates if it does not have it.  Launch it
    via register_gpu_power_task.bat, or from the "Register power-limit helper"
    button in the app's Settings tab (which elevates via ShellExecuteW and so
    arrives here already admin).

    The task has no trigger — it only ever runs on demand.  Re-running this
    script is safe; it overwrites any existing registration.
#>

$ErrorActionPreference = 'Stop'

$TaskName    = 'SynesthesiaAIVideoDirectorGpuPower'
$SourcePs1   = Join-Path $PSScriptRoot 'gpu_power.ps1'
$InstallDir  = Join-Path $env:ProgramData 'Synesthesia AI Video Director\GpuPowerHelper'
$ExchangeDir = Join-Path $env:ProgramData 'Synesthesia AI Video Director\GpuPowerExchange'
$Ps1         = Join-Path $InstallDir 'gpu_power.ps1'

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    # Elevation lives here rather than in the .bat: expressing a quoted path
    # through batch -> powershell -> Start-Process mangles the backslashes.
    # Note the path is pre-quoted into a single -ArgumentList element —
    # Start-Process joins the array on spaces without quoting anything, so an
    # unquoted "H:\reimagine animator\..." would arrive split in two.
    Write-Host 'Requesting administrator rights...'
    try {
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit',
            '-File', ('"{0}"' -f $PSCommandPath)
        )
    } catch {
        Write-Error 'Administrator prompt was declined.'
        exit 1
    }
    exit 0
}
if (-not (Test-Path -LiteralPath $SourcePs1)) {
    Write-Error "Cannot find $SourcePs1."
    exit 1
}

try {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    & icacls.exe $InstallDir '/inheritance:r' '/grant:r' `
        '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' `
        '*S-1-5-11:(OI)(CI)RX' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not protect the installed helper ACL.' }
    Copy-Item -LiteralPath $SourcePs1 -Destination $Ps1 -Force
    & icacls.exe $Ps1 '/inheritance:r' '/grant:r' `
        '*S-1-5-18:F' '*S-1-5-32-544:F' '*S-1-5-11:RX' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not protect the installed helper file.' }

    New-Item -ItemType Directory -Path $ExchangeDir -Force | Out-Null
    & icacls.exe $ExchangeDir '/inheritance:r' '/grant:r' `
        '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' `
        '*S-1-5-11:(OI)(CI)M' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not configure the exchange-directory ACL.' }

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Ps1`""

    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' `
        -LogonType ServiceAccount -RunLevel Highest

    # Queue rather than IgnoreNew: an apply immediately followed by a restore
    # must not have the second run silently dropped.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances Queue `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

    Register-ScheduledTask -TaskName $TaskName -Action $action `
        -Principal $principal -Settings $settings `
        -Description 'Applies and restores NVIDIA GPU power limits for Synesthesia AI Video Director.' `
        -Force | Out-Null

    # THE WHOLE POINT OF THE FEATURE DEPENDS ON THIS.  A task registered with a
    # SYSTEM principal inherits an ACL that grants only Administrators and
    # SYSTEM, so the unelevated app gets "Access is denied" from both
    # `schtasks /query` and `schtasks /run` — the task exists but is useless to
    # it.  Grant Authenticated Users read + execute so it can be triggered
    # without elevation.  Register-ScheduledTask cannot express this, hence the
    # COM drop-down.
    #
    # Read+execute only: nobody but an admin can change what the task runs.  The
    # payload (gpu_power.ps1) re-validates and clamps every value against
    # nvidia-smi's own limits, so the worst another local user can do by
    # triggering it is set a legal power limit on a local GPU.
    $sddl = 'D:(A;;GA;;;BA)(A;;GA;;;SY)(A;;GRGX;;;AU)'
    $svc  = New-Object -ComObject 'Schedule.Service'
    $svc.Connect()
    $svc.GetFolder('\').GetTask($TaskName).SetSecurityDescriptor($sddl, 0)

    Write-Host "Registered scheduled task '$TaskName'."
    Write-Host "  Runs: $Ps1"
    Write-Host '  As:   SYSTEM (highest privileges), on demand only.'
    Write-Host '  ACL:  Authenticated Users may run it (read+execute).'
    exit 0
}
catch {
    Write-Error "Registration failed: $($_.Exception.Message)"
    exit 1
}
