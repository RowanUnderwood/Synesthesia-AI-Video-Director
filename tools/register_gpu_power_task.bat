@echo off
REM  register_gpu_power_task.bat - setup for Image Animator and Video Extender.
REM
REM  Thin launcher for register_gpu_power_task.ps1, which self-elevates.  Double-
REM  click this, accept the single UAC prompt, and the app can thereafter cap and
REM  restore GPU power limits with no further prompts.
REM
REM  The Settings tab's "Register power-limit helper" button runs the .ps1
REM  directly, so this file is only needed for manual setup.
REM
REM  KEEP THIS FILE CRLF-TERMINATED.  cmd.exe seeks batch files by byte offset
REM  and splits tokens mid-word on LF-only line endings ("REM" -> "M"), which
REM  makes the failure look like a syntax error somewhere else entirely.
REM  .gitattributes pins the ending.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0register_gpu_power_task.ps1"
exit /b %ERRORLEVEL%
