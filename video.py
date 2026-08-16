import os
import re
import glob
import json
import time
import random
import shutil
import subprocess
import threading

import requests
import pandas as pd
from pydub import AudioSegment

import config
from models import LLMBridge, sync_video_directory

# Global cache for ffprobe frame counts to speed up preview loading in Tab 3
FRAME_COUNT_CACHE = {}

_zimage_url_cache = None  # Cached after first successful discovery


def _ltx_headers() -> dict:
    """Return Authorization header if an LTX auth token is configured, else empty dict."""
    token = config.LTX_AUTH_TOKEN.strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _mask_token(value):
    """Mask a Bearer token for safe logging, showing first/last 4 chars."""
    prefix = "Bearer "
    if value.startswith(prefix):
        token = value[len(prefix):]
        if len(token) > 8:
            return f"Bearer {token[:4]}...{token[-4:]}"
    return value


def _log_ltx_request(method, url, headers, payload=None):
    """Print full LTX API request details to terminal for debugging."""
    masked = {k: (_mask_token(v) if "authorization" in k.lower() else v) for k, v in headers.items()}
    print(f"\n{'='*60}")
    print(f"LTX API REQUEST")
    print(f"  Method:  {method.upper()}")
    print(f"  URL:     {url}")
    print(f"  Headers: {masked if masked else '(none)'}")
    if payload:
        print(f"  Payload:\n{json.dumps(payload, indent=4)}")
    print(f"{'='*60}")


def _log_ltx_error(method, url, error, response=None):
    """Print full LTX API error details to terminal."""
    print(f"\n{'='*60}")
    print(f"LTX API ERROR")
    print(f"  Method: {method.upper()}")
    print(f"  URL:    {url}")
    print(f"  Error:  {error}")
    if response is not None:
        print(f"  Status: {response.status_code}")
        print(f"  Body:   {response.text}")
    print(f"{'='*60}")


def convert_prompt_for_zimage(base_prompt, pm, settings=None):
    """Convert a video prompt into a still-image first-frame prompt via LLM.

    Uses the project's zimage_prompt_template setting (falls back to the default
    template in config.py). Operates on the raw base_prompt before any style/director
    assembly so the result can be stably cached in the CSV.
    """
    if settings is None:
        settings = pm.load_project_settings()
    template = settings.get("zimage_prompt_template", config.DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE)
    llm_model = config.LM_STUDIO_MODEL
    llm = LLMBridge()
    user_msg = template.replace("{prompt}", base_prompt)
    return llm.query(config.ZIMAGE_PROMPT_SYSTEM_PROMPT, user_msg, llm_model)


def resolve_style_data(style_name, pm):
    """Resolve style data from name, respecting per-project overrides and Custom style."""
    if not style_name or style_name == "None":
        return None
    settings = pm.load_project_settings()
    if style_name == "Custom":
        p = settings.get("custom_style_prompt", "")
        n = settings.get("custom_style_negative", "")
        return {"name": "Custom", "prompt": p, "negative_prompt": n} if p else None
    overrides = settings.get("style_overrides", {})
    if style_name in overrides:
        base = next((s for s in config.STYLES if s["name"] == style_name), {})
        return {
            "name": style_name,
            "prompt": overrides[style_name].get("prompt", base.get("prompt", "{prompt}")),
            "negative_prompt": overrides[style_name].get("negative_prompt", base.get("negative_prompt", "")),
        }
    return next((s for s in config.STYLES if s["name"] == style_name), None)


def apply_character_bibles(prompt, character_bibles):
    """Replace the first occurrence of each character's name with 'name (description)'.
    Subsequent occurrences within the same prompt are left as the plain name,
    preventing the description from appearing more than once per prompt."""
    for name, description in character_bibles.items():
        pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE | re.UNICODE)
        replacement = f"{name} ({description})"
        prompt = pattern.sub(replacement, prompt, count=1)
    return prompt

# ==========================================
# LOGIC: VIDEO GENERATION (LTX)
# ==========================================

def get_project_renders(pm):
    """Get list of rendered final videos with thumbnails for gallery display."""
    if not pm.current_project:
        return [], []

    renders_dir = pm.get_path("renders")
    if not os.path.exists(renders_dir):
        return [], []

    files = sorted(glob.glob(os.path.join(renders_dir, "*.mp4")), key=os.path.getmtime, reverse=True)
    if not files:
        return [], []

    gallery_data = []
    render_paths = []
    for f in files:
        fname = os.path.basename(f)
        try:
            thumb_path = os.path.join(renders_dir, f"thumb_{fname}.jpg")
            if not os.path.exists(thumb_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", f, "-ss", "1", "-vframes", "1", "-q:v", "5", thumb_path],
                    capture_output=True, timeout=10
                )
            if os.path.exists(thumb_path):
                gallery_data.append((thumb_path, fname))
                render_paths.append(f)
            # Skip renders whose thumbnail can't be generated yet (e.g. file still being written)
        except Exception:
            pass  # Skip unreadable files rather than appending None to gallery

    return gallery_data, render_paths

def get_project_videos(pm, project_name=None):
    proj = project_name if project_name else pm.current_project
    if not proj: return []

    vid_dir = os.path.join(pm.base_dir, proj, "videos")
    if not os.path.exists(vid_dir): return []

    files = []
    for extension in ("*.mp4", "*.webm", "*.mov", "*.mkv"):
        files.extend(glob.glob(os.path.join(vid_dir, extension)))

    def sort_key(filepath):
        fname = os.path.basename(filepath)
        parts = fname.split("_")
        shot_id = parts[0].upper() if len(parts) > 0 else fname
        return (shot_id, filepath)

    files = sorted(files, key=sort_key)
    gallery_data = []

    for f in files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        caption = f"{parts[0]}" if len(parts) >= 2 else fname

        if f in FRAME_COUNT_CACHE:
            caption = FRAME_COUNT_CACHE[f]
        else:
            try:
                cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                    "-of", "default=nokey=1:noprint_wrappers=1", f
                ]
                output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
                if output and output.isdigit():
                    caption = f"{caption} ({output} frames)"
                    FRAME_COUNT_CACHE[f] = caption
                else:
                    caption = f"{caption} (? frames)"
                    # not cached — transient failure, retry on next call
            except Exception:
                caption = f"{caption} (? frames)"
                # not cached — transient failure (e.g. file locked by ffprobe), retry on next call

        gallery_data.append((f, caption))

    return gallery_data

def delete_video_file(path, project_name, pm):
    if not path or not os.path.exists(path):
        return get_project_videos(pm, project_name), None
    try:
        os.remove(path)
        if path in FRAME_COUNT_CACHE:
            del FRAME_COUNT_CACHE[path]
        sync_video_directory(pm)
    except Exception as e:
        print(f"Error deleting file: {e}")
    return get_project_videos(pm, project_name), None

def get_video_count_for_shot(shot_id, vid_list):
    count = 0
    for path, caption in vid_list:
        if os.path.basename(path).upper().startswith(f"{str(shot_id).upper()}_"):
            count += 1
    return count

def _discover_zimage_url():
    """Discover the Z-Image endpoint URL.
    For LTX2.3-Multifunctional the endpoint is fixed at /api/generate-image.
    For LTX Desktop, queries /openapi.json and finds the route whose requestBody
    references GenerateImageRequest. Caches the result after first success.
    Returns None on failure."""
    global _zimage_url_cache

    # Extract host root: strip trailing /api or /api/ from LTX_BASE_URL
    base = config.LTX_BASE_URL.rstrip('/')
    host = base[:-4] if base.endswith('/api') else base  # e.g. http://127.0.0.1:8000

    # LTX2.3-Multifunctional has a fixed, known endpoint — skip OpenAPI discovery
    if config.VIDEO_BACKEND == "LTX2.3-Multifunctional":
        return f"{host}/api/generate-image"

    if _zimage_url_cache:
        return _zimage_url_cache

    try:
        _url = f"{host}/openapi.json"
        _hdrs = _ltx_headers()
        _log_ltx_request("GET", _url, _hdrs)
        resp = requests.get(_url, timeout=5, headers=_hdrs)
        resp.raise_for_status()
        schema = resp.json()
        for path, methods in schema.get("paths", {}).items():
            if "post" in methods:
                req_ref = str(methods["post"].get("requestBody", {}))
                if "GenerateImageRequest" in req_ref:
                    _zimage_url_cache = f"{host}{path}"
                    print(f"🔍 Z-Image endpoint discovered: {_zimage_url_cache}")
                    return _zimage_url_cache
    except Exception as e:
        _log_ltx_error("GET", f"{host}/openapi.json", e, getattr(e, "response", None))
        print(f"⚠️ Z-Image endpoint discovery failed: {e}")

    return None


def generate_zimage_first_frame(prompt, shot_id, pm):
    """Call Z-image endpoint, save result to first_frames/, yield string progress updates,
    and yield a final (path, error) tuple as the last item."""
    payload = {
        "prompt": prompt,
        "width": config.Z_IMAGE_WIDTH,
        "height": config.Z_IMAGE_HEIGHT,
        "numSteps": 4,
        "numImages": 1,
    }
    result_container = {}

    def worker():
        url = _discover_zimage_url()
        if not url:
            result_container['error'] = "Could not discover Z-Image endpoint from LTX Desktop OpenAPI schema."
            return
        try:
            _hdrs = _ltx_headers()
            _log_ltx_request("POST", url, _hdrs, payload)
            resp = requests.post(url, json=payload, headers=_hdrs)
            resp.raise_for_status()
            result_container['response'] = resp.json()
        except requests.exceptions.RequestException as e:
            _log_ltx_error("POST", url, e, getattr(e, "response", None))
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f" - {e.response.text}"
            result_container['error'] = err_msg

    t = threading.Thread(target=worker)
    t.start()

    while t.is_alive():
        time.sleep(1)
        try:
            prog_resp = requests.get(f"{config.LTX_BASE_URL}/generation/progress", timeout=2, headers=_ltx_headers())
            if prog_resp.status_code == 200:
                data = prog_resp.json()
                yield f"Z-Image: {data.get('status')} | {data.get('phase')} | {data.get('progress')}%"
        except requests.exceptions.RequestException:
            pass

    t.join()

    if 'error' in result_container:
        yield (None, result_container['error'])
        return

    image_paths = result_container['response'].get('image_paths') or []
    if not image_paths or not os.path.exists(image_paths[0]):
        yield (None, "No image path returned.")
        return

    frames_dir = pm.get_path("first_frames")
    os.makedirs(frames_dir, exist_ok=True)
    save_name = f"{shot_id}_frame_v{int(time.time())}.png"
    local_path = os.path.join(frames_dir, save_name)
    shutil.copy(image_paths[0], local_path)
    print(f"🖼️ Z-Image first frame saved: {local_path}")
    yield (local_path, None)


def generate_comfyui_zimage_first_frame(prompt, shot_id, pm, width=None, height=None):
    """Generate a Z-Image first frame via ComfyUI using ZImage_Poster_API.json.
    Same generator interface as generate_zimage_first_frame: yields progress strings,
    final yield is (local_path, None) on success or (None, error_msg) on failure."""

    # Load workflow template
    workflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "comfyapi", "ZImage_Poster_API.json")
    try:
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
    except Exception as e:
        yield (None, f"Could not load ComfyUI workflow: {e}")
        return

    if isinstance(workflow, list) or "nodes" in workflow:
        yield (None, "ZImage_Poster_API.json is in 'Saved' format — re-export from ComfyUI as API format.")
        return

    # Substitute parameters
    workflow["6"]["inputs"]["text"] = str(prompt)
    workflow["57"]["inputs"]["seed"] = random.randint(1, 10 ** 15)
    workflow["61"]["inputs"]["width"] = int(width or config.Z_IMAGE_WIDTH)
    workflow["61"]["inputs"]["height"] = int(height or config.Z_IMAGE_HEIGHT)
    workflow["73"]["inputs"]["filename_prefix"] = f"synesthesia_{shot_id}"

    # Submit to ComfyUI
    submit_url = f"{config.COMFYUI_URL.rstrip('/')}/prompt"
    submit_payload = {"prompt": workflow}
    _log_ltx_request("POST", submit_url, {}, submit_payload)
    try:
        resp = requests.post(submit_url, json=submit_payload, timeout=15)
        resp.raise_for_status()
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            yield (None, f"ComfyUI did not return a prompt_id. Response: {resp.text}")
            return
    except requests.exceptions.RequestException as e:
        _log_ltx_error("POST", submit_url, e, getattr(e, "response", None))
        yield (None, f"ComfyUI submission failed: {e}")
        return

    # Poll /history/{prompt_id} until complete
    history_url = f"{config.COMFYUI_URL.rstrip('/')}/history/{prompt_id}"
    max_polls = 300  # 5-minute timeout at 1s intervals
    for poll_count in range(max_polls):
        time.sleep(1)
        try:
            hist_resp = requests.get(history_url, timeout=5)
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    # Check for error status
                    status = hist[prompt_id].get("status", {})
                    if status.get("status_str") == "error" or status.get("completed") is False:
                        messages = status.get("messages", [])
                        yield (None, f"ComfyUI generation error: {messages}")
                        return
                    outputs = hist[prompt_id].get("outputs", {})
                    if outputs:
                        break
            yield f"Z-Image (ComfyUI): generating... ({poll_count + 1}s)"
        except requests.exceptions.RequestException:
            yield f"Z-Image (ComfyUI): waiting... ({poll_count + 1}s)"
    else:
        yield (None, "ComfyUI Z-Image generation timed out after 5 minutes.")
        return

    # Extract image info from history output (node "73" is SaveImage)
    try:
        img_info = hist[prompt_id]["outputs"]["73"]["images"][0]
    except (KeyError, IndexError) as e:
        yield (None, f"Could not find output image in ComfyUI history: {e}\nOutputs: {outputs}")
        return

    # Download image via /view endpoint
    view_url = f"{config.COMFYUI_URL.rstrip('/')}/view"
    params = {
        "filename": img_info["filename"],
        "subfolder": img_info.get("subfolder", ""),
        "type": img_info.get("type", "output"),
    }
    try:
        img_resp = requests.get(view_url, params=params, timeout=30)
        img_resp.raise_for_status()
        image_bytes = img_resp.content
    except requests.exceptions.RequestException as e:
        _log_ltx_error("GET", view_url, e, getattr(e, "response", None))
        yield (None, f"Failed to download ComfyUI image: {e}")
        return

    # Save to first_frames/
    frames_dir = pm.get_path("first_frames")
    os.makedirs(frames_dir, exist_ok=True)
    save_name = f"{shot_id}_frame_v{int(time.time())}.png"
    local_path = os.path.join(frames_dir, save_name)
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    print(f"🖼️ Z-Image (ComfyUI) first frame saved: {local_path}")
    yield (local_path, None)


def extract_last_frame(video_path: str, output_path: str):
    """Extract and resize the last frame of video_path to 1920x1080, save to output_path.
    Returns output_path on success, None on failure."""
    import subprocess
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
            "-vframes", "1",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-q:v", "1", output_path
        ], capture_output=True, timeout=15)
        if r.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"extract_last_frame error: {e}")
    return None


def extract_frame_at_time(video_path: str, time_sec: float, output_path: str):
    """Extract the frame nearest to time_sec from video_path, resize to 1920x1080, save to output_path.
    Returns output_path on success, None on failure."""
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-ss", f"{time_sec:.6f}", "-i", video_path,
            "-vframes", "1",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-q:v", "1", output_path
        ], capture_output=True, timeout=15)
        if r.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"extract_frame_at_time error: {e}")
    return None


def get_vocal_chain_predecessor_video_path(shot_id: str, pm):
    """Return the Video_Path of the immediately preceding vocal shot in timeline order,
    or None if there is no consecutive vocal predecessor, or its video doesn't exist."""
    try:
        ids = pm.df['Shot_ID'].astype(str).str.upper().tolist()
        pos = ids.index(str(shot_id).upper())
        if pos == 0:
            return None
        prev_row = pm.df.iloc[pos - 1]
        if str(prev_row.get('Type', '')).strip() != 'Vocal':
            return None
        vp = str(prev_row.get('Video_Path', '')).strip()
        if not vp or not os.path.exists(vp):
            return None
        return vp
    except Exception:
        return None


def _vocal_chain_successor_is_vocal(shot_id: str, pm) -> bool:
    """Return True if the immediately following shot in timeline order is Vocal."""
    try:
        ids = pm.df['Shot_ID'].astype(str).str.upper().tolist()
        pos = ids.index(str(shot_id).upper())
        if pos >= len(ids) - 1:
            return False
        return str(pm.df.iloc[pos + 1].get('Type', '')).strip() == 'Vocal'
    except Exception:
        return False


# Per-resolution maximum durations for chain look-ahead extension.
# 540p's 20s limit is user-verified on LTX Desktop; not reflected in get_ltx_frame_count().
_CHAIN_EXT_MAX_DUR = {"1080p": 5, "720p": 10, "540p": 20}
# Downgrade ladder: if at resolution's max, try the next lower tier.
_CHAIN_EXT_DOWNGRADE = {"1080p": "720p", "720p": "540p"}


def _get_chain_extension_resolution(resolution: str, original_dur: int):
    """Return (effective_resolution, can_extend) for the chain look-ahead extension.

    - Under the resolution max: extend at same resolution.
    - At the max: downgrade to the next-lower tier if that allows extension.
    - No downgrade available (540p at 20s): cannot extend → fall back to old method.
    """
    max_dur = _CHAIN_EXT_MAX_DUR.get(resolution, 10)
    if original_dur < max_dur:
        return resolution, True
    downgraded = _CHAIN_EXT_DOWNGRADE.get(resolution)
    if downgraded is None:
        return resolution, False
    downgraded_max = _CHAIN_EXT_MAX_DUR.get(downgraded, 10)
    if original_dur < downgraded_max:
        return downgraded, True
    return resolution, False


def _will_use_chain_conditioning(shot_id: str, vocal_chain_mode: bool, pm) -> bool:
    """Return True if this shot will receive chain imagePath conditioning from its predecessor,
    meaning it will NOT need a Z-image first frame. Happens when vocal_chain_mode is on and
    both this shot and its direct predecessor are Vocal type."""
    if not vocal_chain_mode:
        return False
    try:
        row = pm.df[pm.df['Shot_ID'] == shot_id]
        if row.empty or str(row.iloc[0].get('Type', '')).strip() != 'Vocal':
            return False
        ids = pm.df['Shot_ID'].astype(str).str.upper().tolist()
        pos = ids.index(str(shot_id).upper())
        if pos == 0:
            return False
        return str(pm.df.iloc[pos - 1].get('Type', '')).strip() == 'Vocal'
    except Exception:
        return False


def _has_valid_cached_first_frame(item: dict, pm) -> bool:
    """Return True if this queue item's shot has a valid cached first-frame image that will be
    reused at render time (caching_mode == 'Use cached image' AND file exists), making
    run-ahead Z-image generation redundant."""
    if item.get('caching_mode') != 'Use cached image':
        return False
    try:
        import pandas as pd
        if 'First_Frame_Image_Path' not in pm.df.columns:
            return False
        row = pm.df[pm.df['Shot_ID'] == item['shot_id']]
        if row.empty:
            return False
        cached_rel = row.iloc[0].get('First_Frame_Image_Path', '')
        if not cached_rel or pd.isna(cached_rel) or not str(cached_rel).strip():
            return False
        abs_path = os.path.join(pm.base_dir, pm.current_project, str(cached_rel).strip())
        return os.path.isfile(abs_path)
    except Exception:
        return False


def generate_video_for_shot(shot_id, resolution, vocal_mode, pm, style=None, director=None, generation_mode="LTX-Native", camera_motion="none", use_llm_image_prompt=False, caching_mode="Use cached prompt", vocal_chain_mode=False, zimage_backend="LTX Desktop", lora_path=None):
    reuse_first_frame = (caching_mode == "Use cached image")
    skip_prompt_cache = (caching_mode == "Regenerate both on each render")
    row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
    if not row_idx:
        yield None, "Error: Shot not found in timeline."
        return

    row = pm.df.loc[row_idx[0]]

    is_override = str(row.get('Prompt_Override', '')).strip().lower() == 'true'
    negative_prompt = config.DEFAULT_NEGATIVE_PROMPT
    style_data = resolve_style_data(style, pm)
    if style_data:
        negative_prompt = config.DEFAULT_NEGATIVE_PROMPT + ", " + style_data["negative_prompt"]

    if is_override:
        vid_prompt = str(row.get('Prompt_Override_Text', '')).strip()
        if not vid_prompt:
            yield None, f"Error: Override flag is set for {shot_id} but override text is empty."
            return
        print(f"⚡ Prompt override active for {shot_id}.")
        print(f"🎬 Override Prompt:\n{vid_prompt}\n=================================\n")
    else:
        vid_prompt_raw = row.get('Video_Prompt', '')
        vid_prompt = "" if pd.isna(vid_prompt_raw) else str(vid_prompt_raw).strip()

        if row.get('Type') == "Vocal" and vocal_mode == "Use Singer/Band Description":
            settings = pm.load_project_settings()
            perf_desc = settings.get("performance_desc", "")
            if perf_desc:
                vid_prompt = perf_desc

        if not vid_prompt:
            yield None, "Error: Missing Video Prompt."
            return

        # Inject character bible descriptions (first occurrence of each name only)
        if pm.character_bibles:
            vid_prompt = apply_character_bibles(vid_prompt, pm.character_bibles)

        if style_data:
            vid_prompt = style_data["prompt"].replace("{prompt}", vid_prompt)

        if director and director != "None":
            effective_director = director
            if director == "Custom":
                settings = pm.load_project_settings()
                effective_director = settings.get("custom_director", "")
            if effective_director:
                vid_prompt += f". This video was directed by {effective_director}."

    print(f"\n🎬 === START VIDEO GENERATION (LTX) ===")
    print(f"🎬 Shot ID: {shot_id} | Type: {row['Type']}")
    if style_data:
        print(f"🎨 Style: {style}")
    print(f"🎬 Video Prompt:\n{vid_prompt}\n=================================\n")

    if config.VIDEO_BACKEND == "MiniMax H3":
        from h3 import generate_h3_video_for_shot
        yield from generate_h3_video_for_shot(
            shot_id, row, row_idx[0], pm, vid_prompt,
            generation_mode=generation_mode,
            caching_mode=caching_mode,
            use_llm_image_prompt=use_llm_image_prompt,
        )
        return

    payload = {
        "prompt": vid_prompt,
        "negativePrompt": negative_prompt,
        "model": "fast",
        "resolution": resolution,
        "aspectRatio": "16:9",
        "duration": int(float(row['Duration'])),
        "fps": 24,
        "cameraMotion": camera_motion,
        "audio": False
    }

    # LoRA support (LTX2.3-Multifunctional only)
    if config.VIDEO_BACKEND == "LTX2.3-Multifunctional" and lora_path and lora_path != "None":
        payload["loraPath"] = lora_path
        payload["loraStrength"] = 1.0

    # --- Vocal Chain: extend duration by 1s to produce a look-ahead chain frame ---
    # Generating 1 extra second allows us to extract a frame just past the shot's intended
    # end to use as imagePath conditioning for the next shot, avoiding a duplicate first frame.
    # If the shot is at its resolution's max duration, downgrade resolution to enable extension.
    _original_dur = int(float(payload["duration"]))
    _extend_for_chain = False
    if vocal_chain_mode and row.get('Type') == 'Vocal':
        if _vocal_chain_successor_is_vocal(shot_id, pm):
            _chain_res, _can_extend = _get_chain_extension_resolution(resolution, _original_dur)
            if _can_extend:
                payload["duration"] = _original_dur + 1
                _extend_for_chain = True
                if _chain_res != resolution:
                    payload["resolution"] = _chain_res
                    yield None, (f"⬇️ Downgrading to {_chain_res} for chain look-ahead "
                                 f"({resolution} is at its {_original_dur}s max).")
                print(f"🔗 Extending {shot_id} to {_original_dur + 1}s"
                      f"{' at ' + _chain_res if _chain_res != resolution else ''} for chain look-ahead.")
            else:
                yield None, (f"⚠️ {shot_id} is at max duration for all resolutions — "
                             f"using fallback chain method (first frame may be near-duplicate).")

    # --- Vocal Chain Last Frame conditioning ---
    _chain_set_imagepath = False
    if vocal_chain_mode and row.get('Type') == 'Vocal':
        try:
            ids = pm.df['Shot_ID'].astype(str).str.upper().tolist()
            pos = ids.index(str(shot_id).upper())
            if pos > 0:
                prev_row = pm.df.iloc[pos - 1]
                if str(prev_row.get('Type', '')).strip() == 'Vocal':
                    pred_id = str(prev_row['Shot_ID'])
                    pred_vid = str(prev_row.get('Video_Path', '')).strip()

                    # Prefer the look-ahead chain frame (avoids duplicate first frame)
                    chain_out_path = os.path.join(
                        pm.get_path("first_frames"),
                        f"{pred_id}_chain_out.jpg"
                    )
                    if os.path.exists(chain_out_path):
                        # Stale check: chain_out.jpg must be newer than the predecessor's video.
                        # If shot N was re-rendered without chain mode, the old chain_out.jpg
                        # no longer matches its visual content.
                        _chain_out_fresh = True
                        if pred_vid and os.path.exists(pred_vid):
                            try:
                                _chain_out_fresh = (os.path.getmtime(chain_out_path)
                                                    >= os.path.getmtime(pred_vid))
                            except OSError:
                                pass  # Can't stat — assume fresh
                        if _chain_out_fresh:
                            payload["imagePath"] = os.path.abspath(chain_out_path)
                            _chain_set_imagepath = True
                            yield None, "🔗 Using look-ahead chain frame from preceding vocal shot..."
                        else:
                            print(f"⚠️ {pred_id}_chain_out.jpg is stale (older than its video) — falling back.")
                            # Fall through to extract_last_frame below
                    if not _chain_set_imagepath and pred_vid and os.path.exists(pred_vid):
                        # Fallback: extract last frame (may produce near-duplicate first frame)
                        _chain_frame_path = os.path.join(
                            pm.get_path("first_frames"),
                            f"{shot_id}_chain_frame_v{int(time.time())}.jpg"
                        )
                        yield None, "🔗 Extracting last frame from preceding vocal shot (no look-ahead available)..."
                        _extracted = extract_last_frame(pred_vid, _chain_frame_path)
                        if _extracted:
                            payload["imagePath"] = os.path.abspath(_extracted)
                            _chain_set_imagepath = True
                        else:
                            yield None, "⚠️ Chain frame extraction failed — continuing without first frame."
        except Exception as e:
            print(f"Vocal chain lookup error: {e}")

    # --- Z-Image first frame conditioning ---
    # Skip if chain already provided imagePath (chain takes precedence for non-first shots in sequence)
    if generation_mode == "Z-Image First Frame" and not _chain_set_imagepath:
        zimage_prompt = vid_prompt  # default: use fully-assembled video prompt as-is

        if use_llm_image_prompt:
            settings_z = pm.load_project_settings()
            # Check per-shot cache in CSV (skipped when caching mode is "Regenerate both")
            cached_ffp = ""
            if not skip_prompt_cache and "First_Frame_Prompt" in pm.df.columns:
                _raw_cached = pm.df.loc[row_idx[0], "First_Frame_Prompt"]
                if _raw_cached and not pd.isna(_raw_cached) and str(_raw_cached).strip():
                    cached_ffp = str(_raw_cached).strip()
            if cached_ffp:
                zimage_prompt = cached_ffp
                yield None, "♻️ Using cached first-frame image prompt..."
            else:
                # For Vocal + "Use Singer/Band Description": use project-level cache
                if row.get("Type") == "Vocal" and vocal_mode == "Use Singer/Band Description":
                    if (not skip_prompt_cache
                            and settings_z.get("zimage_vocal_source_assembled") == vid_prompt
                            and settings_z.get("zimage_vocal_first_frame_prompt")):
                        zimage_prompt = settings_z["zimage_vocal_first_frame_prompt"]
                        yield None, "♻️ Using cached vocal first-frame image prompt..."
                    else:
                        yield None, "🧠 Converting vocal prompt to still image prompt via LLM..."
                        zimage_prompt = convert_prompt_for_zimage(vid_prompt, pm, settings_z)
                        if not skip_prompt_cache:
                            pm.save_project_settings({
                                "zimage_vocal_first_frame_prompt": zimage_prompt,
                                "zimage_vocal_source_assembled": vid_prompt,
                            })
                else:
                    # Convert fully-assembled vid_prompt (with styles, character bibles, director)
                    # so the LLM sees the same prompt that LTX will use for video generation
                    yield None, "🧠 Converting prompt to still image prompt via LLM..."
                    zimage_prompt = convert_prompt_for_zimage(vid_prompt, pm, settings_z)
                # Cache the converted prompt to CSV for reuse (skipped when caching mode is "Regenerate both")
                if not skip_prompt_cache and "First_Frame_Prompt" in pm.df.columns:
                    with pm.queue_lock:
                        pm.df.at[row_idx[0], "First_Frame_Prompt"] = zimage_prompt
                        pm.save_data()

        # Signal the queue processor to update the First Frame Prompt textbox in the UI
        pm._display_ffp = zimage_prompt

        # --- Reuse cached first frame image if requested ---
        _reused_frame = False
        if reuse_first_frame and "First_Frame_Image_Path" in pm.df.columns:
            _cached_rel = pm.df.loc[row_idx[0], "First_Frame_Image_Path"]
            _cached_src = ""
            if "First_Frame_Image_Source" in pm.df.columns:
                _raw_src = pm.df.loc[row_idx[0], "First_Frame_Image_Source"]
                _cached_src = "" if pd.isna(_raw_src) else str(_raw_src)
            if _cached_rel and not pd.isna(_cached_rel) and str(_cached_rel).strip():
                _project_root = os.path.join(pm.base_dir, pm.current_project)
                _cached_abs = os.path.join(_project_root, str(_cached_rel).strip())
                # Only reuse if the file exists AND it was generated from the same assembled prompt
                if os.path.exists(_cached_abs) and _cached_src == zimage_prompt:
                    yield None, "♻️ Reusing cached first frame image..."
                    payload["imagePath"] = os.path.abspath(_cached_abs)
                    _reused_frame = True

        if not _reused_frame:
            print(f"🖼️ === GENERATING Z-IMAGE FIRST FRAME ===")
            print(f"🖼️ Z-Image prompt:\n{zimage_prompt}\n=================================\n")
            frame_path, frame_err = None, "Unknown"

            if zimage_backend == "ComfyUI-run-ahead":
                # Check if background run-ahead thread already produced this shot's frame
                _rh_result = getattr(pm, '_runhead_result', None)
                _rh_thread = getattr(pm, '_runhead_thread', None)
                _rh_shot   = getattr(pm, '_runhead_shot_id', None)

                if _rh_result and _rh_result[0] == shot_id:
                    # Run-ahead already completed before video generation needed it
                    frame_path = _rh_result[1]
                    frame_err = None
                    pm._runhead_result = None
                    yield None, "🚀 Using pre-generated Z-Image first frame (run-ahead)..."
                elif _rh_thread and _rh_shot == shot_id and _rh_thread.is_alive():
                    # Run-ahead in progress — poll-wait while yielding so UI stays responsive
                    _rh_wait_secs = 0
                    yield None, "⏳ Waiting for ComfyUI Z-Image run-ahead to complete..."
                    while _rh_thread.is_alive():
                        time.sleep(1)
                        _rh_wait_secs += 1
                        yield None, f"⏳ Waiting for Z-Image run-ahead... ({_rh_wait_secs}s)"
                    _rh_thread.join()
                    _rh_result = getattr(pm, '_runhead_result', None)
                    if _rh_result and _rh_result[0] == shot_id:
                        frame_path = _rh_result[1]
                        frame_err = None
                        pm._runhead_result = None
                        yield None, "🚀 Using pre-generated Z-Image first frame (run-ahead)..."
                    else:
                        yield None, "⚠️ Z-Image run-ahead failed — generating inline..."

                if frame_path is None:
                    # No run-ahead result available (first shot, single re-render, or run-ahead failed)
                    # Fall back to inline ComfyUI generation
                    _rh_diag_result = getattr(pm, '_runhead_result', None)
                    _rh_diag_running = getattr(pm, '_runhead_running', False)
                    _rh_diag_sid = getattr(pm, '_runhead_shot_id', None)
                    print(f"⚠️ [Run-ahead] Inline fallback for {shot_id}: "
                          f"result={_rh_diag_result}, running={_rh_diag_running}, "
                          f"runhead_shot={_rh_diag_sid}")
                    for _rh_item in generate_comfyui_zimage_first_frame(zimage_prompt, shot_id, pm):
                        if isinstance(_rh_item, tuple):
                            frame_path, frame_err = _rh_item
                        else:
                            yield None, _rh_item
            else:
                # Standard routing: LTX Desktop or ComfyUI (non-run-ahead)
                _zimage_gen = (generate_comfyui_zimage_first_frame if zimage_backend == "ComfyUI"
                               else generate_zimage_first_frame)
                for item in _zimage_gen(zimage_prompt, shot_id, pm):
                    if isinstance(item, tuple):
                        frame_path, frame_err = item
                    else:
                        yield None, item

            if frame_err:
                yield None, f"Error: Z-Image failed: {frame_err}"
                return
            payload["imagePath"] = os.path.abspath(frame_path)
            # Cache the image path and source prompt for future reuse if requested
            if reuse_first_frame and "First_Frame_Image_Path" in pm.df.columns:
                _project_root = os.path.join(pm.base_dir, pm.current_project)
                _rel_path = os.path.relpath(frame_path, _project_root)
                with pm.queue_lock:
                    pm.df.at[row_idx[0], "First_Frame_Image_Path"] = _rel_path
                    if "First_Frame_Image_Source" in pm.df.columns:
                        pm.df.at[row_idx[0], "First_Frame_Image_Source"] = zimage_prompt
                    pm.save_data()

    if row['Type'] == "Vocal":
        vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
        if not vocals_path:
            vocals_path = pm.get_asset_path_if_exists("full_song.mp3")
        if not vocals_path:
            yield None, "Error: Missing audio file for vocal shot. Upload a vocals or full song file."
            return

        try:
            audio = AudioSegment.from_file(vocals_path)
            start_ms = round(float(row['Start_Time']) * 1000)
            end_ms = round(float(row['End_Time']) * 1000)

            chunk = audio[start_ms : end_ms]

            expected_len_ms = end_ms - start_ms
            if len(chunk) < expected_len_ms:
                deficit = expected_len_ms - len(chunk)
                silence_pad = AudioSegment.silent(duration=deficit)
                chunk = chunk + silence_pad

            # Pad audio to match extended duration so LTX accepts the request.
            # Use the next real second of vocals rather than silence so the
            # raw generated clip doesn't appear to end 1 second early.
            if _extend_for_chain:
                extension_chunk = audio[end_ms : end_ms + 1000]
                if len(extension_chunk) < 1000:
                    extension_chunk = extension_chunk + AudioSegment.silent(duration=1000 - len(extension_chunk))
                chunk = chunk + extension_chunk

            chunk_path = os.path.join(pm.get_path("audio_chunks"), f"{shot_id}_audio.mp3")
            chunk.export(chunk_path, format="mp3")

            payload["audio"] = True
            payload["audioPath"] = os.path.abspath(chunk_path)
            payload["model"] = "pro"  # audioPath requires model='pro' in new LTX versions

        except Exception as e:
            print(f"❌ AUDIO ERROR for {shot_id}: {e}")
            yield None, f"Error processing audio: {str(e)}"
            return

    result_container = {}

    def worker():
        try:
            _url = f"{config.LTX_BASE_URL}/generate"
            _hdrs = _ltx_headers()
            # LTX2.3-Multifunctional expects duration/fps/audio as strings
            _send_payload = payload.copy()
            if config.VIDEO_BACKEND == "LTX2.3-Multifunctional":
                _send_payload["duration"] = str(_send_payload["duration"])
                _send_payload["fps"] = str(_send_payload["fps"])
                _send_payload["audio"] = "true" if _send_payload["audio"] else "false"
            else:
                _send_payload = payload
            _log_ltx_request("POST", _url, _hdrs, _send_payload)
            resp = requests.post(_url, json=_send_payload, headers=_hdrs)
            resp.raise_for_status()
            result_container['response'] = resp.json()
        except requests.exceptions.RequestException as e:
            _log_ltx_error("POST", f"{config.LTX_BASE_URL}/generate", e, getattr(e, "response", None))
            err_msg = str(e)
            if e.response is not None:
                err_msg += f" - {e.response.text}"
            result_container['error'] = err_msg

    t = threading.Thread(target=worker)
    t.start()

    while t.is_alive():
        time.sleep(1)
        try:
            prog_resp = requests.get(f"{config.LTX_BASE_URL}/generation/progress", timeout=2, headers=_ltx_headers())
            if prog_resp.status_code == 200:
                data = prog_resp.json()
                status_text = f"LTX Progress - Status: {data.get('status')} | Phase: {data.get('phase')} | {data.get('progress')}%"
                yield None, status_text
        except requests.exceptions.RequestException:
            pass

    t.join()

    if 'error' in result_container:
        print(f"❌ GENERATION FAILED: {result_container['error']}")
        with pm.queue_lock:
            pm.df.at[row_idx[0], 'Status'] = 'Error'
            pm.save_data()
        yield None, f"Error: {result_container['error']}"
        return

    video_path = result_container['response'].get('video_path')
    if video_path and os.path.exists(video_path):
        slug = config.style_to_slug(style) if style and style != "None" else None
        save_name = (
            f"{shot_id}_vid_{slug}_v{int(time.time())}.mp4" if slug
            else f"{shot_id}_vid_v{int(time.time())}.mp4"
        )
        local_path = os.path.join(pm.get_path("videos"), save_name)
        shutil.copy(video_path, local_path)

        with pm.queue_lock:
            pm.df.at[row_idx[0], 'Video_Path'] = local_path
            pm.df.at[row_idx[0], 'Status'] = 'Done'
            pm.df.at[row_idx[0], 'Render_Resolution'] = payload["resolution"]
            pm.save_data()

        if _extend_for_chain:
            # Extract the frame 1 frame past the shot's intended end.
            # LTX uses D*24+1 frames for D seconds, so the last frame of the original
            # duration is at exactly _original_dur seconds; the look-ahead is 1/24s later.
            _chain_out_time = _original_dur + 1 / 24
            _chain_out_path = os.path.join(
                pm.get_path("first_frames"),
                f"{shot_id}_chain_out.jpg"
            )
            yield None, "🔗 Extracting look-ahead chain frame..."
            _extracted = extract_frame_at_time(local_path, _chain_out_time, _chain_out_path)
            if _extracted:
                print(f"🔗 Chain look-ahead frame saved: {_chain_out_path}")
                # Warn if the successor vocal shot is already rendered — its conditioning is now stale.
                try:
                    _s_ids = pm.df['Shot_ID'].astype(str).str.upper().tolist()
                    _s_pos = _s_ids.index(str(shot_id).upper())
                    if _s_pos < len(_s_ids) - 1:
                        _succ_row = pm.df.iloc[_s_pos + 1]
                        if str(_succ_row.get('Type', '')).strip() == 'Vocal':
                            _succ_vid = str(_succ_row.get('Video_Path', '')).strip()
                            if _succ_vid and os.path.exists(_succ_vid):
                                _succ_id = str(_succ_row['Shot_ID'])
                                yield None, (f"⚠️ Shot {_succ_id} is already rendered — "
                                             f"re-render it to apply the new chain frame.")
                except Exception:
                    pass
            else:
                print(f"⚠️ Chain look-ahead frame extraction failed for {shot_id}")

        yield local_path, "Done"
    else:
        pm.df.at[row_idx[0], 'Status'] = 'Error'
        pm.save_data()
        yield None, "Error: Completed but no valid video path returned."
