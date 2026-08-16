@echo off
REM ============================================================
REM  ComfyUI - IMAGE instance (RTX 4090)
REM ============================================================
REM  The text-to-image workflows (ZImage / Ideogram / Krea 2).
REM  Flags mirror what Stability Matrix passes for this package,
REM  plus an explicit --port, so launching from here behaves the
REM  same as pressing Launch inside Stability Matrix.
REM
REM  --cuda-device 0 is the 4090.  This is the CUDA index, which
REM  is NOT nvidia-smi's ordering.  CUDA sees
REM    0 = 4090, 1 = 5090, 2 = 3090
REM
REM  Uses ComfyUI's default user/ dir and database - this is the
REM  primary instance.  Only the video instance
REM  (run_comfy_video.bat) is isolated, so the two processes do
REM  not lock the same database.
REM
REM  Shared model folders keep working because Stability Matrix
REM  has already written extra_model_paths.yaml into this dir.
REM ============================================================

set "COMFY_DIR=H:\Stability Matrix\StabilityMatrix-win-x64\Data\Packages\ComfyUI"

if not exist "%COMFY_DIR%\main.py" (
    echo ERROR: ComfyUI not found at:
    echo   %COMFY_DIR%
    echo Edit COMFY_DIR in this file to match your install.
    pause
    exit /b 1
)

cd /d "%COMFY_DIR%"

echo ============================================================
echo  Starting ComfyUI IMAGE instance  -  port 8188, GPU: 4090
echo ============================================================

venv\Scripts\python.exe main.py ^
    --port 8188 ^
    --cuda-device 0 ^
    --use-sage-attention ^
    --reserve-vram 0.9 ^
    --preview-method auto ^
    --enable-manager

pause
