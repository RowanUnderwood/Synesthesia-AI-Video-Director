# LTX2.3 API Reference

The LTX2.3 backend exposes a REST API on **port 3000**. The web UI (port 4000) talks to this same API, so every feature available in the UI is accessible programmatically.

## Overview

| Property | Value |
|----------|-------|
| Base URL | `http://127.0.0.1:3000` |
| Content-Type | `application/json` (all request bodies) |
| Authentication | None (local only) |
| Prerequisite | Server must be running via `run.bat` |

All responses are JSON unless noted otherwise.

---

## Video Generation

### POST /api/generate

Generate a video from text, image, audio, or keyframe inputs.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Generation prompt (must be non-empty) |
| `model` | string | `"fast"` | `"fast"` \| `"pro"` \| `"ltx-2"` |
| `resolution` | string | `"512p"` | `"512p"` \| `"720p"` \| `"1080p"` |
| `aspectRatio` | string | `"16:9"` | `"16:9"` \| `"9:16"` |
| `duration` | string | `"2"` | Duration in seconds (e.g. `"5"`, `"2.5"`) |
| `fps` | string | `"24"` | Frames per second |
| `cameraMotion` | string | `"none"` | `"none"` \| `"dolly_in"` \| `"dolly_out"` \| `"dolly_left"` \| `"dolly_right"` \| `"jib_up"` \| `"jib_down"` \| `"static"` \| `"focus_shift"` |
| `negativePrompt` | string | `""` | Negative prompt |
| `inferenceSteps` | int | null | Override default step count |
| `imagePath` | string | null | Path to a single conditioning image |
| `startFramePath` | string | null | First frame for image-to-video |
| `endFramePath` | string | null | Last frame for image-to-video |
| `keyframePaths` | string[] | null | Multiple keyframe image paths |
| `keyframeStrengths` | float[] | null | Per-keyframe strength (0.1–1.0) |
| `keyframeTimes` | float[] | null | Time in seconds for each keyframe |
| `audioPath` | string | null | Audio file path for audio-to-video |
| `audio` | string | `"false"` | `"true"` to enable audio generation |
| `loraPath` | string | null | LoRA model file path |
| `loraStrength` | float | `1.0` | LoRA blend strength |
| `modelPath` | string | null | Custom checkpoint file path |

**Response:**

```json
{
  "status": "complete",
  "video_path": "/absolute/path/to/output.mp4",
  "image_paths": []
}
```

`status` is `"complete"` or `"cancelled"`.

> **⚠️ Strict type enforcement:** Unlike LTX Desktop, this API validates field types strictly and returns HTTP 422 if they are wrong. The following fields **must be sent as strings**, not numbers or booleans:
> - `duration` — send `"5"` not `5`
> - `fps` — send `"24"` not `24`
> - `audio` — send `"true"` or `"false"` not `true`/`false`
>
> All other numeric fields (`loraStrength`, `inferenceSteps`, etc.) remain their native types.

---

### GET /api/generation/progress

Poll progress of an active generation. Call repeatedly until `status` is `"complete"` or `"cancelled"`.

**Response:**

```json
{
  "status": "generating",
  "phase": "generating",
  "progress": 45,
  "currentStep": 9,
  "totalSteps": 20
}
```

---

### POST /api/system/cancel

Cancel the currently active generation.

**Response:**

```json
{
  "status": "cancelling",
  "id": "gen-abc123"
}
```

`status` is `"cancelling"` or `"no_active_generation"`.

---

## Image Generation

### POST /api/generate-image

Generate one or more images from a text prompt.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Generation prompt |
| `width` | int | `1024` | Output width in pixels |
| `height` | int | `1024` | Output height in pixels |
| `numSteps` | int | `4` | Inference steps |
| `numImages` | int | `1` | Number of images to generate |

**Response:**

```json
{
  "status": "complete",
  "image_paths": ["/absolute/path/to/image.png"]
}
```

---

## Batch Generation

### POST /api/generate-batch

Generate a multi-segment video by interpolating between consecutive image pairs. Segments are generated individually and concatenated into a single output file.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `segments` | Segment[] | **required** | Array of segments (see below) |
| `resolution` | string | `"720p"` | Output resolution |
| `aspectRatio` | string | `"16:9"` | `"16:9"` \| `"9:16"` |
| `model` | string | `"ltx-2"` | Model to use |
| `fps` | string | `"24"` | Frames per second |
| `audio` | string | `"false"` | Enable audio |
| `cameraMotion` | string | `"static"` | Camera motion for all segments |
| `negativePrompt` | string | `""` | Negative prompt |
| `modelPath` | string | null | Custom checkpoint path |
| `loraPath` | string | null | LoRA file path |
| `loraStrength` | float | `1.0` | LoRA strength |
| `backgroundAudioPath` | string | null | Audio file to mix over final output |

**Segment object:**

```json
{
  "startImage": "/path/to/start.png",
  "endImage": "/path/to/end.png",
  "duration": 3,
  "prompt": "smooth camera pan across the landscape"
}
```

**Response:**

```json
{
  "status": "complete",
  "video_path": "/absolute/path/to/batch_output.mp4"
}
```

---

## Video Enhancement

### POST /api/system/upscale-video

Upscale or re-render a video using SDEdit-based inpainting. Higher `strength` values produce more deviation from the original.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_path` | string | **required** | Path to input video |
| `resolution` | string | `"1080p"` | `"1080p"` \| `"720p"` \| `"544p"` |
| `prompt` | string | `"high quality, detailed, 4k"` | Enhancement prompt |
| `strength` | float | `0.7` | Denoise strength (0.0–1.0) |

**Response:**

```json
{
  "status": "complete",
  "video_path": "/absolute/path/to/upscaled.mp4"
}
```

---

### POST /api/retake

Regenerate a time range within an existing video, optionally replacing audio, video, or both.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_path` | string | **required** | Path to input video |
| `start_time` | float | `0` | Start of retake region (seconds) |
| `duration` | float | `0` | Duration of retake region (seconds) |
| `prompt` | string | `""` | Prompt for the retaken segment |
| `mode` | string | — | `"replace_audio_and_video"` \| `"replace_video"` \| `"replace_audio"` |
| `width` | int | null | Override output width |
| `height` | int | null | Override output height |

**Response:**

```json
{
  "status": "complete",
  "video_path": "/absolute/path/to/retaken.mp4"
}
```

---

## IC-LoRA (Image Conditioning)

### POST /api/ic-lora/extract

Extract a conditioning frame (edge map or depth map) from a video at a specific time.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_path` | string | **required** | Path to video |
| `conditioning_type` | string | `"canny"` | `"canny"` \| `"depth"` |
| `frame_time` | float | `0` | Time in seconds to extract |

**Response:**

```json
{
  "conditioning": "/path/to/canny_frame.png",
  "original": "/path/to/original_frame.png",
  "conditioning_type": "canny",
  "frame_time": 1.5
}
```

---

### POST /api/ic-lora/generate

Generate a video guided by IC-LoRA conditioning images.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_path` | string | **required** | Source video path |
| `conditioning_type` | string | **required** | `"canny"` \| `"depth"` |
| `prompt` | string | **required** | Generation prompt |
| `conditioning_strength` | float | `1.0` | Conditioning influence |
| `num_inference_steps` | int | `30` | Inference steps |
| `cfg_guidance_scale` | float | `1.0` | CFG scale |
| `negative_prompt` | string | `""` | Negative prompt |
| `images` | ImageRef[] | — | Conditioning image references (see below) |

**ImageRef object:**

```json
{
  "path": "/path/to/conditioning_frame.png",
  "frame": 0,
  "strength": 1.0
}
```

**Response:**

```json
{
  "status": "complete",
  "video_path": "/absolute/path/to/output.mp4"
}
```

---

## Prompt Enhancement

### POST /api/suggest-gap-prompt

Use AI to suggest a prompt for a transition or gap between two video segments.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beforePrompt` | string | `""` | Prompt of the preceding segment |
| `afterPrompt` | string | `""` | Prompt of the following segment |
| `beforeFrame` | string | null | Path to last frame of preceding segment |
| `afterFrame` | string | null | Path to first frame of following segment |
| `gapDuration` | float | `5` | Duration of the gap in seconds |
| `mode` | string | `"t2v"` | `"t2v"` \| `"i2v"` |
| `inputImage` | string | null | Conditioning image for i2v mode |

**Response:**

```json
{
  "status": "success",
  "suggested_prompt": "A sweeping aerial shot transitioning..."
}
```

---

## Model Management

### GET /api/models

List available checkpoint files on disk.

**Query params:** `?dir=/custom/path` (optional — uses `models_dir` from settings if omitted)

**Response:**

```json
{
  "models": [
    { "name": "ltx-video-2B.safetensors", "path": "/absolute/path/to/file" }
  ],
  "path": "/scanned/directory",
  "error": null
}
```

---

### GET /api/models/status

Get download and load status for all required model components.

**Response:**

```json
{
  "models": [
    {
      "id": "checkpoint",
      "name": "LTX Video Checkpoint",
      "downloaded": true,
      "size": 5368709120,
      "expected_size": 5368709120,
      "required": true
    }
  ],
  "all_downloaded": true,
  "total_size_gb": 12.4,
  "downloaded_size_gb": 12.4,
  "models_path": "/path/to/models",
  "has_api_key": false,
  "use_local_text_encoder": true
}
```

---

### GET /api/models/required

Get the list of required model type IDs.

**Response:**

```json
{
  "modelTypes": ["checkpoint", "text_encoder"]
}
```

---

### POST /api/models/download

Start downloading model files.

**Request body:**

```json
{
  "modelTypes": ["checkpoint", "text_encoder"]
}
```

**Response:**

```json
{
  "status": "started",
  "message": "Download started",
  "sessionId": "dl-abc123"
}
```

---

### GET /api/models/download-progress

Poll download progress. Pass the `sessionId` from the download start response.

**Query params:** `?sessionId=dl-abc123`

**Response (in progress):**

```json
{
  "status": "downloading",
  "current_downloading_file": "ltx-video-2B.safetensors",
  "current_file_progress": 0.42,
  "total_progress": 0.21,
  "total_downloaded_bytes": 2684354560,
  "expected_total_bytes": 12884901888,
  "completed_files": [],
  "all_files": ["ltx-video-2B.safetensors", "t5xxl_fp8.safetensors"],
  "speed_bytes_per_sec": 52428800
}
```

**Response (complete):** `{ "status": "complete" }`

**Response (error):** `{ "status": "error", "error": "message" }`

---

### POST /api/text-encoder/download

Download the text encoder model.

**Response:**

```json
{
  "status": "started",
  "message": "Download started",
  "sessionId": "dl-xyz789"
}
```

`status` is `"started"` or `"already_downloaded"`.

---

## LoRA Management

### GET /api/loras

List LoRA files in the configured LoRA directory.

**Query params:** `?dir=/custom/path` (optional)

**Response:**

```json
{
  "loras": [
    { "name": "my-style.safetensors", "path": "/absolute/path/to/file" }
  ],
  "loras_dir": "/path/to/loras",
  "error": null
}
```

> **Integration note:** The response is an **object** with a `loras` array, not a plain list. Each entry has `name` (filename for display) and `path` (absolute path to pass as `loraPath` in generation requests). Synesthesia's `_fetch_lora_choices()` in `ui/tab3_video.py` extracts `entry["path"]` from each item in `data["loras"]`.

---

### GET /api/lora-dir

Get the currently saved LoRA directory path.

**Response:**

```json
{
  "loraDir": "/path/to/loras",
  "error": null
}
```

---

### POST /api/lora-dir

Set the LoRA directory path (persisted to `settings.json`).

**Request body:**

```json
{
  "loraDir": "/path/to/loras"
}
```

**Response:**

```json
{
  "status": "ok",
  "loraDir": "/path/to/loras"
}
```

---

## File Operations

### POST /api/system/upload-image

Upload a base64-encoded image to the server. Returns the server-side path for use in generation requests.

**Request body:**

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "filename": "my-frame.png"
}
```

The `data:image/...;base64,` prefix is optional and will be stripped automatically.

**Response:**

```json
{
  "status": "success",
  "path": "/absolute/path/to/my-frame.png"
}
```

---

### GET /api/system/file

Serve a file by absolute path. Use this to retrieve generated videos and images.

**Query params:** `?path=/absolute/path/to/file.mp4`

**Response:** Binary file data (video, image, etc.)

---

### GET /api/system/history

List generated output files with pagination.

**Query params:** `?page=1&limit=20`

**Response:**

```json
{
  "status": "success",
  "history": [
    {
      "filename": "output_20240101_120000.mp4",
      "type": "video",
      "mtime": 1704110400.0,
      "fullpath": "/absolute/path/to/output.mp4"
    }
  ],
  "total_pages": 3,
  "current_page": 1,
  "total_items": 54
}
```

---

### POST /api/system/delete-file

Delete a file from the outputs directory.

**Request body:**

```json
{
  "filename": "output_20240101_120000.mp4"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "File deleted"
}
```

---

## Output Directory

### GET /api/system/get-dir

Get the current output directory path.

**Response:**

```json
{
  "status": "success",
  "directory": "/absolute/path/to/outputs"
}
```

---

### POST /api/system/set-dir

Set a custom output directory (persisted).

**Request body:**

```json
{
  "directory": "C:/Users/me/Videos/ltx-outputs"
}
```

**Response:**

```json
{
  "status": "success",
  "directory": "C:/Users/me/Videos/ltx-outputs"
}
```

---

### GET /api/system/browse-dir

Open a native Windows folder picker dialog. **Blocks until the user makes a selection.** Not suitable for headless use.

**Response:**

```json
{
  "path": "C:/Users/me/Videos"
}
```

`path` is `null` if the user cancelled.

---

## GPU & Memory

### GET /api/gpu-info

Get GPU availability and VRAM information.

**Response:**

```json
{
  "cuda_available": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "vram_gb": 24,
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 4090",
    "vram": 24576,
    "vramUsed": 4096
  }
}
```

---

### GET /api/system/list-gpus

List all CUDA-capable GPU devices.

**Response:**

```json
{
  "status": "success",
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "vram": "24.0 GB",
      "vram_mb": 24576,
      "active": true
    }
  ]
}
```

---

### POST /api/system/switch-gpu

Switch which GPU is used for generation (takes effect on next model load).

**Request body:**

```json
{
  "gpu_id": 1
}
```

**Response:**

```json
{
  "status": "success",
  "gpu_id": 1
}
```

---

### POST /api/system/clear-gpu

Unload all models and release GPU memory immediately.

**Response:**

```json
{
  "status": "success",
  "message": "GPU memory cleared and models unloaded"
}
```

---

### GET /api/system/low-vram-mode

Check whether low-VRAM CPU offload mode is active.

**Response:**

```json
{
  "enabled": false
}
```

---

### POST /api/system/low-vram-mode

Enable or disable low-VRAM CPU offload mode. When enabled, model layers are offloaded to system RAM between inference steps.

**Request body:**

```json
{
  "enabled": true
}
```

**Response:**

```json
{
  "status": "success",
  "enabled": true
}
```

---

## System State

### GET /health

Server health and model status.

**Response:**

```json
{
  "status": "ready",
  "models_loaded": true,
  "active_model": "fast",
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 4090",
    "vram": 24576,
    "vramUsed": 8192
  },
  "sage_attention": false,
  "models_status": [
    { "id": "checkpoint", "name": "LTX Checkpoint", "loaded": true, "downloaded": true }
  ]
}
```

---

### POST /api/system/reset-state

Reset internal generation state flags without unloading models. Use this if the server gets stuck in a bad state after an interrupted generation.

**Response:**

```json
{
  "status": "success",
  "message": "Generation state reset"
}
```

---

### GET /api/runtime-policy

Check whether the server is forcing API-based generation instead of local GPU.

**Response:**

```json
{
  "force_api_generations": false
}
```

If `force_api_generations` is `true`, set `fal_api_key` to `""` in `settings.json` to use local GPU. See `patches/API模式问题修复说明.md` for details.

---

## Settings

### GET /api/settings

Get current application settings.

**Response:** Full settings object (see fields below).

---

### POST /api/settings

Update application settings. Send only the fields you want to change.

**Request body (all fields optional):**

| Field | Type | Description |
|-------|------|-------------|
| `use_torch_compile` | bool | Enable Torch compile optimization |
| `load_on_startup` | bool | Auto-load models when server starts |
| `ltx_api_key` | string | LTX API key (set to dummy value to disable API mode) |
| `user_prefers_ltx_api_video_generations` | bool | Use LTX cloud API instead of local GPU |
| `fal_api_key` | string | FAL API key (empty = disabled, forces local GPU) |
| `use_local_text_encoder` | bool | Use local T5 encoder instead of API |
| `fast_model` | object | `{ "use_upscaler": bool, "steps": int }` |
| `pro_model` | object | `{ "steps": int, "use_upscaler": bool }` |
| `prompt_cache_size` | int | Number of prompts to cache |
| `prompt_enhancer_enabled_t2v` | bool | Auto-enhance prompts for text-to-video |
| `prompt_enhancer_enabled_i2v` | bool | Auto-enhance prompts for image-to-video |
| `gemini_api_key` | string | Gemini API key for prompt enhancement |
| `seed_locked` | bool | Use a fixed seed for all generations |
| `locked_seed` | int | The fixed seed value |
| `models_dir` | string | Custom directory to scan for model checkpoints |
| `lora_dir` | string | Custom directory to scan for LoRA files |

---

## Practical Examples

### Check server is ready

```bash
curl http://127.0.0.1:3000/health
```

### Text-to-video

```bash
curl -X POST http://127.0.0.1:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A red fox running through a snowy forest at dawn",
    "model": "fast",
    "resolution": "720p",
    "duration": "3",
    "fps": "24"
  }'
```

### Image-to-video

```bash
curl -X POST http://127.0.0.1:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The camera slowly pulls back to reveal the full scene",
    "model": "ltx-2",
    "resolution": "720p",
    "duration": "4",
    "startFramePath": "C:/frames/start.png",
    "endFramePath": "C:/frames/end.png"
  }'
```

### Keyframe-anchored video

```bash
curl -X POST http://127.0.0.1:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Epic fantasy landscape transformation",
    "model": "ltx-2",
    "resolution": "720p",
    "duration": "6",
    "keyframePaths": ["C:/kf/frame1.png", "C:/kf/frame2.png", "C:/kf/frame3.png"],
    "keyframeStrengths": [0.9, 0.7, 0.9],
    "keyframeTimes": [0.0, 3.0, 6.0]
  }'
```

### Poll for progress

```bash
# Start generation (returns immediately with progress endpoint available)
# Then poll:
curl http://127.0.0.1:3000/api/generation/progress
# Repeat until "status" is "complete" or "cancelled"
```

### Upload an image, then use it

```bash
# 1. Upload
RESPONSE=$(curl -s -X POST http://127.0.0.1:3000/api/system/upload-image \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w 0 my-image.png)\", \"filename\": \"my-image.png\"}")

# 2. Extract server path
IMAGE_PATH=$(echo $RESPONSE | python -c "import sys,json; print(json.load(sys.stdin)['path'])")

# 3. Use in generation
curl -X POST http://127.0.0.1:3000/api/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"The scene comes to life\", \"imagePath\": \"$IMAGE_PATH\", \"duration\": \"3\"}"
```

### Batch generation (multi-segment)

```bash
curl -X POST http://127.0.0.1:3000/api/generate-batch \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {
        "startImage": "C:/frames/scene1_start.png",
        "endImage": "C:/frames/scene1_end.png",
        "duration": 3,
        "prompt": "Sunrise over mountains, timelapse"
      },
      {
        "startImage": "C:/frames/scene2_start.png",
        "endImage": "C:/frames/scene2_end.png",
        "duration": 4,
        "prompt": "Clouds rolling over the peaks"
      }
    ],
    "resolution": "720p",
    "model": "ltx-2"
  }'
```
