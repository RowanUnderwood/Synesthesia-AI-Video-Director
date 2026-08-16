Here's the full mechanism, in the order you'd port it.

  The problem it solves

  nvidia-smi -pl requires Administrator. So does the NVML equivalent (nvmlDeviceSetPowerManagementLimit) — switching
  libraries doesn't get you out of it. Running your whole app elevated is the naive fix, and it's bad: everything the
  app spawns inherits the token.

  The design here splits privilege: an unelevated app writes a request file, and an on-demand Scheduled Task running as
  SYSTEM consumes it. One UAC prompt, ever — at install time. Zero prompts thereafter.

  The four files

  ┌───────────────────────────────────┬───────────────────────┬──────────────────────────────────────────────────┐
  │               File                │        Runs as        │                       Job                        │
  ├───────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────┤
  │ gpu_power.py                      │ your app (unelevated) │ query, write plan, fire task, poll result        │
  ├───────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────┤
  │ tools/gpu_power.ps1               │ SYSTEM, via the task  │ validate plan, call nvidia-smi -pl, write result │
  ├───────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────┤
  │ tools/register_gpu_power_task.ps1 │ Administrator, once   │ register the task + fix its ACL                  │
  ├───────────────────────────────────┼───────────────────────┼──────────────────────────────────────────────────┤
  │ tools/register_gpu_power_task.bat │ user                  │ thin double-clickable launcher for the above     │
  └───────────────────────────────────┴───────────────────────┴──────────────────────────────────────────────────┘

  Plus two runtime files at the repo root: gpu_power_plan.json (app → helper) and gpu_power_result.json (helper → app).
  Gitignore both.

  The handshake

  gpu_power.py:229 _dispatch() is the whole protocol:

  1. Generate a nonce (uuid4 hex), delete any stale result file.
  2. Write gpu_power_plan.json: {"nonce", "mode": "apply"|"restore", "targets": [{"index", "watts"}]}.
  3. schtasks /run /tn <TaskName>.
  4. Poll gpu_power_result.json every 0.2 s for up to 25 s, accepting it only if its nonce matches.

  Step 4 is not optional ceremony. schtasks /run is fire-and-forget — it returns 0 the instant the scheduler accepts the
  request, telling you nothing about whether the work happened or succeeded. The nonce is the only way to distinguish
  "our run finished" from "a leftover result file from three runs ago."

  Read utf-8-sig on the result (gpu_power.py:261) and write it BOM-less on the PowerShell side (gpu_power.ps1:30) —
  PowerShell's Out-File/Set-Content will happily emit a BOM that json.loads chokes on, hence the explicit
  UTF8Encoding($false).

  Registering the task — the step that's easy to get wrong

  register_gpu_power_task.ps1 self-elevates if needed (:27), then:

  $action    = New-ScheduledTaskAction -Execute 'powershell.exe' `
      -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Ps1`""
  $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' `
      -LogonType ServiceAccount -RunLevel Highest
  $settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
      -DontStopIfGoingOnBatteries -MultipleInstances Queue `
      -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
  Register-ScheduledTask -TaskName $TaskName -Action $action `
      -Principal $principal -Settings $settings -Force

  No trigger — on-demand only. -MultipleInstances Queue matters: an apply immediately followed by a restore must not
  have the second run dropped, which is what the default IgnoreNew would do.

  Then the part the whole feature hinges on (register_gpu_power_task.ps1:81-84):

  $sddl = 'D:(A;;GA;;;BA)(A;;GA;;;SY)(A;;GRGX;;;AU)'
  $svc  = New-Object -ComObject 'Schedule.Service'
  $svc.Connect()
  $svc.GetFolder('\').GetTask($TaskName).SetSecurityDescriptor($sddl, 0)

  A task registered with a SYSTEM principal inherits an ACL granting only Administrators and SYSTEM. Your unelevated app
  then gets "Access is denied" from both schtasks /query and schtasks /run — the task exists and is completely useless
  to you. Register-ScheduledTask has no parameter for this, so you drop to the Schedule.Service COM object. GRGX = read
  + execute for Authenticated Users: they can trigger it, they cannot change what it runs.

  This is also why helper_installed() (gpu_power.py:174) treats "access is denied" as installed. Reporting it as missing
  would send the user to re-run a setup step that already succeeded, when the real problem is a stale ACL from an older
  registration.

  Trust boundary

  The plan file is written unelevated and consumed with SYSTEM rights, so gpu_power.ps1 trusts nothing in it:

  - Indices and wattages are cast [int] (twice — again at :108, right before use).
  - Wattages are clamped against min/max that the script queries from nvidia-smi itself, not from the plan.
  - The command line is assembled from validated integers only; no file content is ever interpolated into it.

  Worst case for a hostile plan file: a legal power limit on a local GPU. Keep this property if you modify it — it's
  what makes the Authenticated-Users ACL safe.

  Restore mode ignores the plan's targets entirely (gpu_power.ps1:88-91) and walks every GPU nvidia-smi reports,
  resetting each to its own power.default_limit. A stale or truncated plan therefore can never leave a card capped.

  After applying, it re-queries and reports applied from the hardware rather than echoing what you asked for.

  Porting checklist

  1. Copy all four files, preserving tools/ as a subdirectory of your app root — gpu_power.ps1:20 does Split-Path
  -Parent $PSScriptRoot to find the plan file, and gpu_power.py:38-43 mirrors that from the Python side.
  2. Rename TASK_NAME (gpu_power.py:45 and register_gpu_power_task.ps1:20) — must match, and must differ from
  ReimagineGpuPower. Register-ScheduledTask -Force overwrites by name; two apps sharing a task name means the last one
  registered wins and the other silently drives the wrong script path.
  3. Rewrite GPU_ROLES and DEFAULT_WATTS (gpu_power.py:50-62) for your hardware. Role keys are what your settings file
  speaks; they map to a substring matched against the nvidia-smi GPU name.
  4. Wire the settings keys — watts_from_settings() expects flat power_watts_<role> keys, reconcile_on_start() expects
  power_limit_mode.
  5. Register once: double-click register_gpu_power_task.bat, accept one UAC prompt.
  6. Verify: python gpu_power.py prints state, helper status, and per-card bounds; python gpu_power.py apply / restore
  are live smoke tests.

  If your app has no UI to elevate from, drop register_helper() (gpu_power.py:191) and just ship the .bat. It's only
  there so the Settings tab can trigger registration via ShellExecuteW(..., "runas", ...) — the sole way to raise
  privileges from an unelevated process, and one that cannot suppress the consent dialog.

  Non-obvious traps

  nvidia-smi's device order is not CUDA's. On this machine nvidia-smi reports 0=5090, 1=4090, 2=3090 while CUDA sees
  0=4090, 1=5090, 2=3090. resolve_roles() (gpu_power.py:122) matches on GPU name, never position. Hardcoding CUDA order
  sends the 5090 a wattage below its 400 W floor (rejected) and the 4090 its own stock limit (no-op) — no power saved
  and no visible error. Whatever your box, resolve by name.
  Trust boundary

  The plan file is written unelevated and consumed with SYSTEM rights, so gpu_power.ps1 trusts nothing in it:

  - Indices and wattages are cast [int] (twice — again at :108, right before use).
  - Wattages are clamped against min/max that the script queries from nvidia-smi itself, not from the plan.
  - The command line is assembled from validated integers only; no file content is ever interpolated into it.

  Worst case for a hostile plan file: a legal power limit on a local GPU. Keep this property if you modify it — it's
  what makes the Authenticated-Users ACL safe.

  Restore mode ignores the plan's targets entirely (gpu_power.ps1:88-91) and walks every GPU nvidia-smi reports,
  resetting each to its own power.default_limit. A stale or truncated plan therefore can never leave a card capped.

  After applying, it re-queries and reports applied from the hardware rather than echoing what you asked for.

  Porting checklist

  1. Copy all four files, preserving tools/ as a subdirectory of your app root — gpu_power.ps1:20 does Split-Path
  -Parent $PSScriptRoot to find the plan file, and gpu_power.py:38-43 mirrors that from the Python side.
  2. Rename TASK_NAME (gpu_power.py:45 and register_gpu_power_task.ps1:20) — must match, and must differ from
  ReimagineGpuPower. Register-ScheduledTask -Force overwrites by name; two apps sharing a task name means the last one
  registered wins and the other silently drives the wrong script path.
  3. Rewrite GPU_ROLES and DEFAULT_WATTS (gpu_power.py:50-62) for your hardware. Role keys are what your settings file
  speaks; they map to a substring matched against the nvidia-smi GPU name.
  4. Wire the settings keys — watts_from_settings() expects flat power_watts_<role> keys, reconcile_on_start() expects
  power_limit_mode.
  5. Register once: double-click register_gpu_power_task.bat, accept one UAC prompt.
  6. Verify: python gpu_power.py prints state, helper status, and per-card bounds; python gpu_power.py apply / restore
  are live smoke tests.

  If your app has no UI to elevate from, drop register_helper() (gpu_power.py:191) and just ship the .bat. It's only
  there so the Settings tab can trigger registration via ShellExecuteW(..., "runas", ...) — the sole way to raise
  privileges from an unelevated process, and one that cannot suppress the consent dialog.

  Non-obvious traps

  nvidia-smi's device order is not CUDA's. On this machine nvidia-smi reports 0=5090, 1=4090, 2=3090 while CUDA sees
  0=4090, 1=5090, 2=3090. resolve_roles() (gpu_power.py:122) matches on GPU name, never position. Hardcoding CUDA order
  sends the 5090 a wattage below its 400 W floor (rejected) and the 4090 its own stock limit (no-op) — no power saved
  and no visible error. Whatever your box, resolve by name.

  Name matching takes the first hit only. resolve_roles breaks on role not in found, so two identical cards resolve to
  one role and only one gets capped. If your other machine has duplicate GPUs, that loop needs reworking — it's the one
  piece of this that isn't machine-agnostic.

  Per-card floors are real and differ wildly — 5090: 400–600 W, 4090: 10–479 W, 3090: 100–385 W. Clamp on the way in
  (_as_watts, app.py:396) so the value you persist matches what lands on the hardware, and clamp again in the helper. A
  rejected -pl is the main silent-failure mode.

  Limits do not survive a reboot (no persistence mode on GeForce under WDDM), but they do survive your process exiting —
  Windows holds an nvidia-smi -pl setting until reboot or a driver reload. That asymmetry drives three separate restore
  paths:

  - explicit shutdown (the only guaranteed one),
  - atexit + SIGINT/SIGTERM/SIGBREAK handlers (app.py:4381) — covers Ctrl-C, not a hard kill or anything calling
  os._exit,
  - reconcile_on_start() (gpu_power.py:340) — the real safety net. At startup, if the setting isn't wattage_cap but a
  card is still below stock, restore. Costs one cheap nvidia-smi query in the normal case and never wakes the task.

  Key both the guards and reconcile off the hardware (is_capped()), not off the setting — that way caps left by a mode
  switch or an earlier crash get cleared too.

  .bat/.ps1 must stay CRLF-terminated. cmd.exe seeks batch files by byte offset and splits tokens mid-word on LF-only
  endings — REM becomes M, and the error names a token that appears nowhere in the file. Pin it in .gitattributes.

  Quoted paths mangle through batch → PowerShell → Start-Process. That's why elevation lives in the .ps1 (:35-38) and
  not the .bat, and why the script path is pre-quoted into a single -ArgumentList element — Start-Process joins the
  array on spaces without quoting, so an unquoted H:\reimagine animator\... arrives split in two.

  As SYSTEM, the user PATH is gone. Get-NvidiaSmi (gpu_power.ps1:35) falls back to System32\nvidia-smi.exe and Program
  Files\NVIDIA Corporation\NVSMI\.

  Nothing in gpu_power.py raises. Every public function returns a value or an (ok, message) pair, because it all runs
  inside UI callbacks where an exception is a dead end. query_gpus() returning [] means "no power-limit support on this
  machine," not an error