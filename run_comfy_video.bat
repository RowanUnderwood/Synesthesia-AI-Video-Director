@echo off
REM ============================================================
REM  ComfyUI - VIDEO instance (RTX 5090)
REM ============================================================
REM  Second ComfyUI process, used only for the MiniMax H3 video
REM  workflow.  The image instance managed by Stability Matrix
REM  stays on :8188 / the 4090; this one takes the 5090 so the
REM  two model stacks never evict each other.
REM
REM  --cuda-device 1 is the 5090.  NOTE: this is the CUDA index,
REM  which is NOT nvidia-smi's ordering.  CUDA sees
REM    0 = 4090, 1 = 5090, 2 = 3090
REM  while nvidia-smi lists the 5090 first.  Verify with:
REM    venv\Scripts\python.exe -c "import torch;print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
REM
REM  models/, custom_nodes/, input/ and output/ stay shared with
REM  the image instance - that is deliberate, so an uploaded
REM  frame and the rendered clip live in one place.  Only user/
REM  and the SQLite DB are separated, so the two processes do not
REM  lock the same database.
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
if not exist "user_video" mkdir "user_video"

echo ============================================================
echo  Starting ComfyUI VIDEO instance  -  port 8189, GPU: 5090
echo ============================================================

venv\Scripts\python.exe main.py ^
    --port 8189 ^
    --cuda-device 1 ^
    --use-sage-attention ^
    --reserve-vram 0.9 ^
    --user-directory "%COMFY_DIR%\user_video" ^
    --database-url "sqlite:///user_video/comfyui_video.db"

pause
