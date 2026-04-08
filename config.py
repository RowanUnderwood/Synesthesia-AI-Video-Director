import os
import json
import re
import glob

# ==========================================
# CONFIGURATION
# ==========================================
LTX_BASE_URL = "http://127.0.0.1:8000/api"
LTX_AUTH_TOKEN: str = ""  # Bearer token for LTX Desktop auth; empty = no auth (vanilla LTX)
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
LLM_PROVIDER = "LM Studio"  # "LM Studio" | "MiniMax"
MINIMAX_API_KEY = ""
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODELS = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
VIDEO_BACKEND = "LTX Desktop"  # "LTX Desktop" | "Wan2GP"
ELECTRICITY_COST = 0.1805  # USD per kWh (default 18.05¢)
SYSTEM_WATTAGE = 600.0     # Watts, full system draw during generation (default: RTX 5090 system)
GPU_MONITOR_INDEX = 0      # pynvml device index to monitor for VRAM usage
VRAM_WARN_THRESHOLD = 0.92 # Warn if dedicated VRAM usage > 92%
SLOWDOWN_WARN_FACTOR = 2.5 # Warn if actual render took > 2.5x estimated time

# pynvml state — initialized once on first call
_nvml_initialized = False
_nvml_available = False
_nvml_handle_cache = {}  # {device_index: handle}

def _ensure_nvml():
    global _nvml_initialized, _nvml_available
    if _nvml_initialized:
        return _nvml_available
    _nvml_initialized = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_available = True
    except Exception:
        _nvml_available = False
    return _nvml_available

def get_vram_usage():
    """Returns (used_gb, total_gb) tuple or None if pynvml unavailable or fails."""
    if not _ensure_nvml():
        return None
    try:
        import pynvml
        if GPU_MONITOR_INDEX not in _nvml_handle_cache:
            _nvml_handle_cache[GPU_MONITOR_INDEX] = pynvml.nvmlDeviceGetHandleByIndex(GPU_MONITOR_INDEX)
        info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle_cache[GPU_MONITOR_INDEX])
        return (info.used / 1024**3, info.total / 1024**3)
    except Exception:
        _nvml_handle_cache.pop(GPU_MONITOR_INDEX, None)  # Evict stale handle on error
        return None

def get_gpu_list():
    """Returns list of 'index — Name' strings for the Tab 5 GPU selector dropdown."""
    if not _ensure_nvml():
        return ["0 — (pynvml unavailable)"]
    try:
        import pynvml
        count = pynvml.nvmlDeviceGetCount()
        result = []
        for i in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                result.append(f"{i} — {name}")
            except Exception:
                result.append(f"{i} — (unknown)")
        return result if result else ["0 — (no GPUs found)"]
    except Exception:
        return ["0 — (pynvml error)"]

# RTX 5090 baseline: estimated render seconds per second of output video (LTX-Native only).
RENDER_TIME_PER_SEC = {
    "1080p": {"LTX-Native": 14.89},
    "720p":  {"LTX-Native":  9.75},
    "540p":  {"LTX-Native":  8.20},
}

# RTX 5090 baseline: fixed overhead in seconds for Z-Image first-frame generation.
# This is a constant cost per shot regardless of clip duration (generates one 1920x1080 still).
Z_IMAGE_OVERHEAD_SECS = {
    "1080p": 20.0,
    "720p":  15.0,
    "540p":  12.0,
}

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_calibration.json")
CALIBRATION_MAX_SAMPLES = 15

def get_calibrated_rate(resolution, generation_mode="LTX-Native"):
    """Return calibrated render rate (seconds per second of video) for LTX-Native video generation.
    Uses rolling average of recorded actuals; falls back to RENDER_TIME_PER_SEC baseline."""
    key = f"{resolution}|LTX-Native"
    try:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            samples = data.get("samples", {}).get(key, [])
            if samples:
                return sum(samples) / len(samples)
    except Exception:
        pass
    table = RENDER_TIME_PER_SEC.get(resolution, RENDER_TIME_PER_SEC["720p"])
    return table["LTX-Native"]

def get_calibrated_zimage_overhead(resolution):
    """Return calibrated Z-Image fixed overhead in seconds.
    Uses rolling average of recorded actuals; falls back to Z_IMAGE_OVERHEAD_SECS baseline."""
    key = f"{resolution}|ZImageOverhead"
    try:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            samples = data.get("samples", {}).get(key, [])
            if samples:
                return sum(samples) / len(samples)
    except Exception:
        pass
    return Z_IMAGE_OVERHEAD_SECS.get(resolution, 15.0)

def record_render_time(resolution, generation_mode, actual_duration_secs, actual_render_secs):
    """Record a completed render for self-calibration. Silently ignores errors.

    For LTX-Native: records seconds-per-second-of-video rate.
    For Z-Image First Frame: records the inferred fixed overhead (total time minus expected
    video generation time) separately, so the two costs calibrate independently."""
    if actual_duration_secs <= 0 or actual_render_secs <= 0:
        return
    try:
        data = {"version": 1, "samples": {}}
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
        samples = data.get("samples", {})

        if generation_mode == "Z-Image First Frame":
            # Infer Z-image overhead = total time minus expected video generation portion.
            # Use the current native rate estimate (calibrated or baseline).
            native_rate = get_calibrated_rate(resolution, "LTX-Native")
            inferred_overhead = actual_render_secs - (native_rate * actual_duration_secs)
            if inferred_overhead > 0:
                key = f"{resolution}|ZImageOverhead"
                key_samples = samples.get(key, [])
                key_samples.append(round(inferred_overhead, 2))
                if len(key_samples) > CALIBRATION_MAX_SAMPLES:
                    key_samples = key_samples[-CALIBRATION_MAX_SAMPLES:]
                samples[key] = key_samples
        else:
            rate = actual_render_secs / actual_duration_secs
            key = f"{resolution}|LTX-Native"
            key_samples = samples.get(key, [])
            key_samples.append(round(rate, 4))
            if len(key_samples) > CALIBRATION_MAX_SAMPLES:
                key_samples = key_samples[-CALIBRATION_MAX_SAMPLES:]
            samples[key] = key_samples

        data["samples"] = samples
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def reset_render_calibration():
    """Delete the calibration file, reverting estimates to 5090 baseline."""
    try:
        if os.path.exists(CALIBRATION_FILE):
            os.remove(CALIBRATION_FILE)
        return "✅ Calibration reset to 5090 baseline."
    except Exception as e:
        return f"❌ Error resetting calibration: {e}"

def estimate_render_seconds(duration_secs, resolution, generation_mode):
    """Estimate render time using calibrated values if available, else 5090 baseline.

    Uses an additive model for Z-Image First Frame: native video generation time plus
    a fixed per-shot overhead for the image generation step (which is constant regardless
    of clip duration)."""
    native_rate = get_calibrated_rate(resolution)
    est = float(duration_secs) * native_rate
    if generation_mode == "Z-Image First Frame":
        est += get_calibrated_zimage_overhead(resolution)
    return est

Z_IMAGE_WIDTH = 1920
Z_IMAGE_HEIGHT = 1080

REQUIRED_COLUMNS = [
    "Shot_ID", "Type",
    "Start_Time", "End_Time", "Duration",
    "Start_Frame", "End_Frame", "Total_Frames",
    "Lyrics", "Video_Prompt", "First_Frame_Prompt", "First_Frame_Image_Path", "First_Frame_Image_Source", "Prompt_Override", "Prompt_Override_Text", "Characters", "Video_Path", "All_Video_Paths", "Status",
    "Render_Resolution",
]

RESOLUTION_MAP = {
    "540p": (960, 540),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}

DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, artifacts, watermark"

def load_styles():
    styles_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.json")
    try:
        with open(styles_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

STYLES = load_styles()
STYLE_NAMES = ["None"] + [s["name"] for s in STYLES] + ["Custom"]

CAMERA_MOTIONS = [
    "none", "static", "focus_shift",
    "dolly_in", "dolly_out",
    "dolly_left", "dolly_right",
    "jib_up", "jib_down",
]

DIRECTORS = [
    "None", "Custom",
    # American New Wave / Contemporary
    "Wes Anderson", "David Lynch", "Quentin Tarantino", "Tim Burton",
    "Terrence Malick", "Stanley Kubrick", "Edgar Wright", "Christopher Nolan",
    "Sofia Coppola", "Darren Aronofsky", "Spike Jonze", "Michel Gondry",
    # International Masters
    "Yasujirō Ozu", "Akira Kurosawa", "Federico Fellini", "Wong Kar-wai",
    "Park Chan-wook", "Denis Villeneuve", "Guillermo del Toro",
    # Additional distinctive voices
    "Jean-Luc Godard", "Werner Herzog", "David Fincher", "Paul Thomas Anderson",
]

def style_to_slug(style_name):
    """Convert a style display name to a filename-safe slug.
    'LTX - Claymation' → 'claymation', 'LTX - Film Noir' → 'film_noir'"""
    if not style_name or style_name == "None":
        return None
    name = re.sub(r'^LTX\s*-\s*', '', style_name, flags=re.IGNORECASE)
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

def slug_to_style_name(slug):
    """Reverse lookup slug → display name. Falls back to slug if not found."""
    for s in STYLES:
        if style_to_slug(s["name"]) == slug:
            return s["name"]
    return slug

def slug_from_filename(fname):
    """Extract style slug from a video filename, or None if unstyled.
    'S001_vid_claymation_v1726524982.mp4' → 'claymation'
    'S001_vid_v1726524982.mp4' → None
    """
    match = re.match(r'^[^_]+_vid_(.+)_v\d+\.mp4$', fname)
    return match.group(1) if match else None

def get_styles_in_videos_dir(pm):
    """Scan the videos/ dir and return sorted list of style display names present."""
    vid_dir = pm.get_path("videos")
    if not vid_dir or not os.path.exists(vid_dir):
        return []
    slugs = set()
    for f in glob.glob(os.path.join(vid_dir, "*.mp4")):
        slug = slug_from_filename(os.path.basename(f))
        if slug:
            slugs.add(slug)
    return sorted([slug_to_style_name(s) for s in slugs])

DEFAULT_CONCEPT_PROMPT = (
    "Context: The overarching plot is: {plot}\n"
    "Previous Shot Visual: {prev_shot}\n"
    "Current Shot Info: Timestamp {start}s, Duration {duration}s, Type: {type}.\n"
    "Task: Write a highly detailed visual description for this video shot encompassing the action. "
    "Describe the scene, camera motion, emotions, lighting and performance only. "
    "Pay special attention to the camera's motion. Do not include any additional notes or titles."
)

LTX_SYSTEM_PROMPT = """You are an expert cinematography AI director writing video generation prompts. Adhere strictly to these rules:

1. Establish the shot: Use cinematography terms that match the preferred film genre. Include aspects like scale or specific category characteristics.
2. Set the scene: Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.
3. Describe the action: Write the core action as a natural sequence, flowing from beginning to end.
4. Reference your character(s): Refer to recurring characters by their first name only. Do not describe their physical appearance — visual consistency is handled separately by the character bible.
5. Identify camera movement(s): Specify when the view should shift and how. Include how subjects or objects appear after the camera motion.
6. Format: Keep your prompt in a SINGLE flowing paragraph.
7. Grammar: Use present tense verbs to describe movement and action.
8. Detail scale: Match your detail to the shot scale (Closeups need more precise detail than wide shots).
9. Camera focus: When describing camera movement, focus on the camera's relationship to the subject.
10. Length: Write 4 to 8 descriptive sentences to cover all key aspects."""

DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE = (
    "Redesign this video prompt to be an image prompt representing the first frame of the video. "
    "It is ok to keep descriptions of camera position, type, and usage but remove anything about passage of time or motion. "
    "If there are multiple actions described, describe only the first one. "
    "Keep your prompt as close to the original as possible while following these guidelines. "
    "There should be no text overlays mention about the image. "
    "Return the new image prompt only, no other text. "
    "For example, do not include a sentence at the end of the image prompt listing the correctly followed instructions :D "
    "This will just be misinterpreted by the image model AS additional instructions. "
    "Thanks! \n\nVideo prompt: {prompt}"
)
ZIMAGE_PROMPT_SYSTEM_PROMPT = "You are an expert still image prompt writer for AI image generators."

SCRIPTED_PROMPT_TEMPLATE = """Create a short narrative film via AI video prompts while adhering to the following main character, gender, settings, and rough concept.  See also the following csv shot list with durations and frame counts.  Return the shot list csv data with each "Video_Prompt" field filled out. include the Shot_ID and Type fields for these rows.  Do NOT include any other text in your reply.  Enclose the video prompt column in "" to prevent any commas inside the video prompt from corrupting the data.
Follow the ltx prompt guide below to create each "action" prompt. Give each recurring character a unique first name and refer to them ONLY by that first name throughout — do NOT describe their physical appearance in the prompts. A separate character bible will inject visual descriptions automatically to keep characters consistent.
1. Establish the Shot
Use cinematography terms that match your intended genre. Include shot scale or category-specific characteristics to refine the visual style.
2. Set the Scene
Describe lighting conditions, color palette, surface textures, and atmosphere to establish mood and tone.
3. Describe the Action
Write the core action as a natural sequence, flowing clearly from beginning to end.
4. Reference the Character(s)
Use each character's assigned first name. Do not describe their physical appearance — visual descriptions are handled by the character bible.
5. Identify Camera Movement(s)
Specify how and when the camera moves. Describing how subjects appear after the movement helps the model complete the motion accurately.
6. Describe the Audio
Clearly describe ambient sound, music, speech, or singing.
Place spoken dialogue in quotation marks
Specify language and accent if needed
For Best Results
Write your prompt as a single flowing paragraph
Use present tense verbs for action and movement
Match the level of detail to the shot scale
(close-ups need more detail than wide shots)
Describe camera movement relative to the subject
Aim for 4-8 descriptive sentences

Main Character's Gender: {gender}
Main Character and Setting Description: {character_desc}
Rough Concept: {concept}

Shot list:
{shot_list}"""

ALL_VOCALS_PROMPT_TEMPLATE = """Create a music video via AI video prompts for the following song (see song lyrics below).  See the attached CSV formatted shot list with durations and frame counts. Every shot in this video is a vocal/performance shot. Create detailed, visually compelling prompts for each shot that combine performance elements with creative visual storytelling aligned to the song's themes. Do not include any guns in the story as the LTX video model censors them.  Do not use any words in your descriptions like, painted, sketched, or drawn to prevent the video model from creating animated shots.  Return the shot list in CSV format with just the "Shot_ID", "Type" and "Video_Prompt" columns.  Do NOT include any other text in your reply.  Enclose the video prompt column in "" to prevent any commas inside the video prompt from corrupting the data.

Follow the ltx prompt guide below to create each prompt. Give each recurring character a unique first name and refer to them ONLY by that first name throughout — do NOT describe their physical appearance in the prompts. A separate character bible will inject visual descriptions automatically to keep characters consistent.

Establish the shot. Use cinematography terms that match your preferred film genre. Include aspects like scale or specific category characteristics to further refine the style you're looking for.

Set the scene. Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.

Describe the action. Write the core action as a natural sequence, flowing from beginning to end.

Reference your character(s) by their assigned first name. Do not describe their physical appearance in the prompt.

Identify camera movement(s). Specify when the view should shift and how. Including how subjects or objects appear after the camera motion gives the model a better idea of how to finish the motion.

Keep your prompt in a single flowing paragraph to give the model a cohesive scene to work with.
Use present tense verbs to describe movement and action.
Match your detail to the shot scale. Closeups need more precise detail than wide shots.
When describing camera movement, focus on the camera's relationship to the subject.
You should expect to write 4 to 8 descriptive sentences to cover all the key aspects of the prompt.

Song Lyrics:
{lyrics}

User suggested plot concept:
{plot}

Singer/Band/Venue Description:
{performance_desc}

Shot list:
{shot_list}"""

BULK_PROMPT_TEMPLATE = """Create a music video via AI video prompts for the following song (see song lyrics below).  See the attached CSV formatted shot list with durations and frame counts. We need to tell a coherent story using the shots labeled "Action" in the type column.  Align your story loosely to the themes and metaphors present in the song's lyrics, or the user suggested plot concept (if present), but do not be afraid to get creative! Do not include any guns in the story as the LTX video model censors them.  Do not use any words in your descriptions like, painted, sketched, or drawn to prevent the video model from creating animated shots.  Return the shot list in CSV format with just the "Shot_ID", "Type" and "Video_Prompt" columns.  Leave the "Vocal" type rows video prompts blank, but include the Shot_ID and Type fields for these rows.  Do NOT include any other text in your reply.  Enclose the video prompt column in "" to prevent any commas inside the video prompt from corrupting the data.

Follow the ltx prompt guide below to create each "action" prompt. Give each recurring character a unique first name and refer to them ONLY by that first name throughout — do NOT describe their physical appearance in the prompts. A separate character bible will inject visual descriptions automatically to keep characters consistent.

Establish the shot. Use cinematography terms that match your preferred film genre. Include aspects like scale or specific category characteristics to further refine the style you're looking for.

Set the scene. Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.

Describe the action. Write the core action as a natural sequence, flowing from beginning to end.

Reference your character(s) by their assigned first name. Do not describe their physical appearance in the prompt.

Identify camera movement(s). Specify when the view should shift and how. Including how subjects or objects appear after the camera motion gives the model a better idea of how to finish the motion.

Keep your prompt in a single flowing paragraph to give the model a cohesive scene to work with.
Use present tense verbs to describe movement and action.
Match your detail to the shot scale. Closeups need more precise detail than wide shots.
When describing camera movement, focus on the camera's relationship to the subject.
You should expect to write 4 to 8 descriptive sentences to cover all the key aspects of the prompt.

Song Lyrics:
{lyrics}

User suggested plot concept:
{plot}

Shot list:
{shot_list}"""

# ==========================================
# CHARACTER BIBLE TEMPLATES
# ==========================================

CHARACTER_BIBLE_SYSTEM_PROMPT = "You are a casting director. Only output valid CSV data with no additional text."

CHARACTER_BIBLE_USER_TEMPLATE = (
    "The following are video shot descriptions for a story.\n"
    "Identify all recurring named characters (those appearing by first name in two or more shots).\n"
    "Return a CSV with exactly two columns: character_name, description\n\n"
    "For each character write a complete physical description covering: gender, approximate age, "
    "hair color and style, eye color, distinguishing features, jewelry, makeup, and clothing/style.\n\n"
    "CRITICAL: If any physical detail is not mentioned in the story you MUST invent a specific vivid "
    "value for it — never write 'not specified', 'unknown', 'none mentioned', or similar placeholders. "
    "Every detail must be a concrete visual trait an AI video model can render. "
    "Invent a hair color, an eye color, a clothing style, etc. as needed.\n\n"
    "Format each description as a compact comma-separated list of visual traits. "
    "Enclose the description column value in double quotes.\n"
    "Output ONLY the CSV header row followed by data rows. No other text.\n\n"
    "Shot descriptions:\n"
    "{shot_prompts}"
)

# ==========================================
# DEFAULT LLM PROMPT TEMPLATES
# ==========================================

# --- Plot Generation ---
DEFAULT_PLOT_SYSTEM_PROMPT_MUSIC = "You are a creative writer for music videos."
DEFAULT_PLOT_USER_TEMPLATE_MUSIC = (
    "Rough Concept: {concept}\n\nLyrics:\n{lyrics}\n\nTimeline:\n{timeline}\n\n"
    "Task: Write a cohesive linear plot summary for this video (max 300 words)."
)
DEFAULT_PLOT_SYSTEM_PROMPT_SCRIPTED = "You are a creative writer for short narrative films."
DEFAULT_PLOT_USER_TEMPLATE_SCRIPTED = (
    "Rough Concept: {concept}\n\n"
    "Task: Write a cohesive linear plot summary for this short narrative film (max 300 words). "
    "Focus on character arcs, settings, and dramatic moments."
)

# --- Performance Description ---
DEFAULT_PERF_SYSTEM_PROMPT_MUSIC = "You are a casting director and set designer."
DEFAULT_PERF_USER_TEMPLATE_MUSIC = (
    "Concept: {concept}\nPlot: {plot}\n{gender_instruction}\n"
    "Task: Invent and describe the physical appearance, proper name, age, hair, clothing, and style of a lead singer for the above song, specifically for an AI video generation model. "
    "Start with the phrase 'Handheld dynamic closeup shot of'. Do not include any details about the character that would be out of view in a close-up shot. "
    "End the description with; [name of singer] is careful to enunciate each word to the camera to account for their deaf sister's lip reading. "
    "Shot with a dynamic camera movement and slight handheld shake, shallow depth of field, dramatic chiaroscuro lighting, 85mm lens, 24fps, high contrast, crowd silhouettes, energetic atmosphere, cinematic color grading, [describe color pallet here] "
    "Keep it concise (5-6 sentences)."
)
DEFAULT_PERF_SYSTEM_PROMPT_SCRIPTED = "You are a casting director and set designer for short films."
DEFAULT_PERF_USER_TEMPLATE_SCRIPTED = (
    "Concept: {concept}\nPlot: {plot}\n{gender_instruction}\n"
    "Task: Describe the main character's physical appearance, style, and the primary setting/location, "
    "specifically for an AI video generation model. Include build, face, hair, clothing, and setting details. "
    "Keep it concise (3-4 sentences)."
)

# ==========================================
# GLOBAL DEFAULTS — keys that can be promoted to global and seeded into new projects
# ==========================================

GLOBALIZABLE_KEYS = frozenset({
    # Prompt templates
    "plot_sys_prompt_music", "plot_user_template_music",
    "plot_sys_prompt_scripted", "plot_user_template_scripted",
    "perf_sys_prompt_music", "perf_user_template_music",
    "perf_sys_prompt_scripted", "perf_user_template_scripted",
    "concepts_bulk_template", "concepts_vocals_template", "concepts_scripted_template",
    "bible_sys_prompt", "bible_user_template",
    "prompt_template", "zimage_prompt_template",
    # Timeline defaults
    "min_silence", "silence_thresh", "shot_mode", "min_dur", "max_dur", "video_mode",
    # Video generation preferences
    "firstframe_mode", "llm_image_prompt_mode", "first_frame_reuse_mode",
    "vocal_prompt_mode", "vocal_chain_mode", "last_resolution", "last_versions",
    "last_camera_motion", "last_director", "last_style",
})

_CODE_DEFAULTS = {
    "plot_sys_prompt_music": DEFAULT_PLOT_SYSTEM_PROMPT_MUSIC,
    "plot_user_template_music": DEFAULT_PLOT_USER_TEMPLATE_MUSIC,
    "plot_sys_prompt_scripted": DEFAULT_PLOT_SYSTEM_PROMPT_SCRIPTED,
    "plot_user_template_scripted": DEFAULT_PLOT_USER_TEMPLATE_SCRIPTED,
    "perf_sys_prompt_music": DEFAULT_PERF_SYSTEM_PROMPT_MUSIC,
    "perf_user_template_music": DEFAULT_PERF_USER_TEMPLATE_MUSIC,
    "perf_sys_prompt_scripted": DEFAULT_PERF_SYSTEM_PROMPT_SCRIPTED,
    "perf_user_template_scripted": DEFAULT_PERF_USER_TEMPLATE_SCRIPTED,
    "concepts_bulk_template": BULK_PROMPT_TEMPLATE,
    "concepts_vocals_template": ALL_VOCALS_PROMPT_TEMPLATE,
    "concepts_scripted_template": SCRIPTED_PROMPT_TEMPLATE,
    "bible_sys_prompt": CHARACTER_BIBLE_SYSTEM_PROMPT,
    "bible_user_template": CHARACTER_BIBLE_USER_TEMPLATE,
    "prompt_template": DEFAULT_CONCEPT_PROMPT,
    "zimage_prompt_template": DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE,
    "min_silence": 700,
    "silence_thresh": -45,
    "shot_mode": "Random",
    "min_dur": 2,
    "max_dur": 4,
    "video_mode": "Intercut",
    "firstframe_mode": "LTX-Native",
    "llm_image_prompt_mode": "Use video prompt as-is",
    "first_frame_reuse_mode": "Use cached prompt",
    "vocal_prompt_mode": "Use Singer/Band Description",
    "vocal_chain_mode": False,
    "last_resolution": "1080p",
    "last_versions": 1,
    "last_camera_motion": "none",
    "last_director": "None",
    "last_style": "None",
}

# ==========================================
# GLOBAL MEMORY
# ==========================================
GLOBAL_SETTINGS_FILE = "global_settings.json"

def get_global_llm():
    try:
        if os.path.exists(GLOBAL_SETTINGS_FILE):
            with open(GLOBAL_SETTINGS_FILE, "r") as f:
                return json.load(f).get("last_llm", None)
    except:
        pass
    return None

def save_global_llm(model_id):
    try:
        data = {}
        if os.path.exists(GLOBAL_SETTINGS_FILE):
            with open(GLOBAL_SETTINGS_FILE, "r") as f:
                data = json.load(f)
        data["last_llm"] = model_id
        with open(GLOBAL_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except:
        pass

def load_global_url_settings():
    global LTX_BASE_URL, LTX_AUTH_TOKEN, LM_STUDIO_URL, LLM_PROVIDER, MINIMAX_API_KEY, VIDEO_BACKEND, ELECTRICITY_COST, SYSTEM_WATTAGE, GPU_MONITOR_INDEX
    try:
        if os.path.exists(GLOBAL_SETTINGS_FILE):
            with open(GLOBAL_SETTINGS_FILE, "r") as f:
                data = json.load(f)
                LTX_BASE_URL = data.get("ltx_base_url", LTX_BASE_URL)
                LTX_AUTH_TOKEN = data.get("ltx_auth_token", LTX_AUTH_TOKEN)
                LM_STUDIO_URL = data.get("lm_studio_url", LM_STUDIO_URL)
                LLM_PROVIDER = data.get("llm_provider", LLM_PROVIDER)
                MINIMAX_API_KEY = data.get("minimax_api_key", MINIMAX_API_KEY)
                VIDEO_BACKEND = data.get("video_backend", VIDEO_BACKEND)
                ELECTRICITY_COST = float(data.get("electricity_cost", ELECTRICITY_COST))
                SYSTEM_WATTAGE = float(data.get("system_wattage", SYSTEM_WATTAGE))
                GPU_MONITOR_INDEX = int(data.get("gpu_monitor_index", GPU_MONITOR_INDEX))
    except:
        pass

def save_global_url_settings(settings: dict):
    """Accept a settings dict and persist all global settings to disk."""
    global LTX_BASE_URL, LTX_AUTH_TOKEN, LM_STUDIO_URL, LLM_PROVIDER, MINIMAX_API_KEY, VIDEO_BACKEND, ELECTRICITY_COST, SYSTEM_WATTAGE, GPU_MONITOR_INDEX
    LTX_BASE_URL = str(settings.get("ltx_base_url", LTX_BASE_URL)).strip()
    LTX_AUTH_TOKEN = str(settings.get("ltx_auth_token", LTX_AUTH_TOKEN)).strip()
    LM_STUDIO_URL = str(settings.get("lm_studio_url", LM_STUDIO_URL)).strip()
    LLM_PROVIDER = settings.get("llm_provider", LLM_PROVIDER)
    MINIMAX_API_KEY = str(settings.get("minimax_api_key", MINIMAX_API_KEY)).strip()
    VIDEO_BACKEND = settings.get("video_backend", VIDEO_BACKEND)
    ELECTRICITY_COST = float(settings.get("electricity_cost", ELECTRICITY_COST))
    SYSTEM_WATTAGE = float(settings.get("system_wattage", SYSTEM_WATTAGE))
    raw_gpu = settings.get("gpu_monitor_index", GPU_MONITOR_INDEX)
    GPU_MONITOR_INDEX = int(str(raw_gpu).split(" — ")[0]) if raw_gpu is not None else 0
    try:
        data = {}
        if os.path.exists(GLOBAL_SETTINGS_FILE):
            with open(GLOBAL_SETTINGS_FILE, "r") as f:
                data = json.load(f)
        data.update({
            "ltx_base_url": LTX_BASE_URL,
            "ltx_auth_token": LTX_AUTH_TOKEN,
            "lm_studio_url": LM_STUDIO_URL,
            "llm_provider": LLM_PROVIDER,
            "minimax_api_key": MINIMAX_API_KEY,
            "video_backend": VIDEO_BACKEND,
            "electricity_cost": ELECTRICITY_COST,
            "system_wattage": SYSTEM_WATTAGE,
            "gpu_monitor_index": GPU_MONITOR_INDEX,
        })
        for key in GLOBALIZABLE_KEYS:
            if key in settings:
                data[key] = settings[key]
        with open(GLOBAL_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return "✅ Settings saved and applied."
    except Exception as e:
        return f"❌ Error saving settings: {e}"


def get_global_defaults() -> dict:
    """Return all GLOBALIZABLE_KEYS with values from global_settings.json,
    falling back to hardcoded _CODE_DEFAULTS for any key not yet customised."""
    result = dict(_CODE_DEFAULTS)
    try:
        if os.path.exists(GLOBAL_SETTINGS_FILE):
            with open(GLOBAL_SETTINGS_FILE, "r") as f:
                data = json.load(f)
            for key in GLOBALIZABLE_KEYS:
                if key in data:
                    result[key] = data[key]
    except Exception:
        pass
    return result


def get_calibration_summary():
    """Return human-readable calibration stats string for display in Tab 5."""
    try:
        if not os.path.exists(CALIBRATION_FILE):
            return "No calibration data yet. Run some renders to build up accuracy."
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        samples = data.get("samples", {})
        if not samples:
            return "No calibration samples recorded."
        lines = []
        for key in sorted(samples.keys()):
            s = samples[key]
            if s:
                avg = sum(s) / len(s)
                res, mode = key.split("|", 1)
                if mode == "ZImageOverhead":
                    baseline = Z_IMAGE_OVERHEAD_SECS.get(res, 15.0)
                    pct = (avg / baseline * 100) if baseline > 0 else 0
                    lines.append(f"{key}: {len(s)} samples, avg {avg:.1f}s fixed overhead  ({pct:.0f}% of 5090 baseline)")
                else:
                    baseline_table = RENDER_TIME_PER_SEC.get(res, RENDER_TIME_PER_SEC["720p"])
                    baseline = baseline_table.get(mode, baseline_table["LTX-Native"])
                    pct = (avg / baseline * 100) if baseline > 0 else 0
                    lines.append(f"{key}: {len(s)} samples, avg {avg:.2f}s/s  ({pct:.0f}% of 5090 baseline)")
        return "\n".join(lines) if lines else "No calibration samples recorded."
    except Exception as e:
        return f"Error reading calibration data: {e}"

load_global_url_settings()
