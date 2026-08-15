# Synesthesia AI Video Director — Project Context

## Architecture

Modular Gradio application split across `app.py` (entry point) and supporting modules:
- **`models.py`** — `ProjectManager` (project I/O, CSV, assets) and `LLMBridge` (LM Studio API)
- **`config.py`** — API endpoints, resolution map, LLM prompt templates, style loading
- **`timeline.py`** — Audio silence analysis, shot generation, LTX frame locking
- **`llm_logic.py`** — Prompt/plot generation orchestration, LLM response parsing
- **`video.py`** — LTX video generation per shot, gallery display, frame count cache
- **`assembly.py`** — moviepy video assembly, cutting room floor compilation
- **`utils.py`** — Frame snapping, base64 image encoding, restart hotkey
- **`ui/`** — Gradio UI split into 6 tabs (project, storyboard, video, assembly, settings, help)

Orchestrates:
- **LM Studio** (local LLM) — generates video prompts and plot summaries via OpenAI-compatible API
- **LTX Desktop** (local AI video engine) — generates video clips from prompts
- **moviepy 1.x** — assembles clips into final video with audio

## Critical: LTX Desktop Resolution Handling

LTX Desktop generates videos at resolutions that are **multiples of 32** for optimal GPU processing. The actual output resolutions do NOT match standard video resolutions, and they **vary depending on whether audio is attached** to the clip (Vocal vs Action shots produce different resolutions at the same preset). For example, 540p without audio = 960x512, but 540p with audio = 960x576.

Because LTX output resolutions are unpredictable, `RESOLUTION_MAP` uses standard resolutions (for UI labels and API requests only). The `assemble_video` function **dynamically detects** the target resolution by reading the first available video clip's actual dimensions. All other clips are resized to match. Do NOT hardcode LTX output resolutions.

## Dependencies

- **moviepy must be < 2.0** — the codebase uses `from moviepy.editor import ...` which was removed in moviepy 2.x. Version is pinned in `requirements.txt`.
- **pydub** requires FFmpeg installed on the system PATH.
- **keyboard** is used for the Ctrl+R restart hotkey.
- **`styles.json`** — optional file in the project root that defines named prompt style presets; loaded at startup by `config.py`.

## Settings Persistence — Rules for New Features

Synesthesia uses a **two-tier settings model**. Every user-adjustable control must fit into one of these tiers; session-only controls should be the rare exception.

### Tier 1 — Global defaults (`global_settings.json`)

Stored by `config.save_global_url_settings()`. Loaded at startup by `config.load_global_url_settings()` into module-level globals. Seeded into every new project's `settings.json` via `config.get_global_defaults()` in `models.create_project()`.

**What belongs here:** API endpoints, wattage/cost settings, any setting defined in `config.GLOBALIZABLE_KEYS` that has been promoted by the user via "📌 Make Current Project Settings Default" (Tab 5).

`config.GLOBALIZABLE_KEYS` is the authoritative whitelist of settings that *can* become global defaults. `config._CODE_DEFAULTS` maps every key in that frozenset to its hardcoded fallback constant.

### Tier 2 — Project settings (`projects/<name>/settings.json`)

Stored by `pm.save_project_settings(dict)` and read by `pm.load_project_settings()`. Each project's file starts as a copy of the current global defaults (seeded on creation) and can diverge freely per project.

**What belongs here:** Everything the user adjusts in Tab 2 (timeline settings, prompt templates, concept/plot), Tab 3 generation preferences (resolution, first-frame mode, style, director, etc.), and any other per-project state.

**What does NOT belong here:** Purely ephemeral UI navigation state (e.g. which shot is selected in a gallery scroll).

### Adding a new UI control — checklist

1. **Decide the tier.** Ask: does this setting make sense as a new-project default? If yes, add its key to `GLOBALIZABLE_KEYS` and its code-level fallback to `_CODE_DEFAULTS` in `config.py`.

2. **Auto-save on change.** Wire a `.change()` (dropdowns/sliders/radios) or `.blur()` (textboxes) handler that calls `pm.save_project_settings({"key": value})`. Tab 2 uses `auto_save_tab2` with a shared inputs list; Tab 3 uses `auto_save_tab3_prefs`. Add new controls to the appropriate function's inputs list and settings dict.

3. **Restore on project load.** Add the value to `handle_load()`'s return tuple in `ui/app.py` using `settings.get("key", fallback)`, and add the corresponding component to the `outputs=[...]` list of `t1["load_btn"].click()`. **The return tuple length must exactly match the outputs list length — Gradio raises a silent error if they differ.**

4. **Restore on project create.** If `handle_create()` resets or populates this control, add it to `handle_create()`'s return tuple and `t1["create_btn"].click(outputs=[...])` in the same position as in `handle_load`.

5. **Visibility side effects.** If loading a saved value should also change the visibility of other components (e.g. Z-Image sub-controls appearing when `firstframe_mode = "Z-Image First Frame"`), handle this explicitly in `handle_load`'s return tuple with `gr.update(visible=..., value=...)`. Gradio does **not** fire `.change()` events when `handle_load` sets a value programmatically.

### Project-content fields — never globalise

These keys live in `project/settings.json` only and must **never** be added to `GLOBALIZABLE_KEYS`:
`rough_concept`, `plot`, `performance_desc`, `singer_gender`, `total_time_spent`, `scripted_total_dur`, `scripted_shot_count`, `style_overrides`, `custom_style_prompt`, `custom_style_negative`, `custom_director`, `zimage_vocal_first_frame_prompt`, `zimage_vocal_source_assembled`, `last_ffp_style`, `last_ffp_director`, `last_gen_mode`.

---

## Key Domain Concepts

- **Shot Types**: "Vocal" (singing/performance) and "Action" (narrative/visual). These control prompt generation strategy and audio attachment during video generation.
- **LTX duration snapping**: Shot durations are locked to whole-second increments at 24 fps (LTX generates `D*24+1` frames for a D-second clip, e.g. 4s = 97 frames). Maximum duration is 10s; 1080p is further capped at 5s. Shots over 5s are automatically downgraded to 720p by `_effective_resolution()` in `ui/tab3_video.py`.
- **Per-resolution duration limits**: 1080p max 5s, 720p max 10s, 540p max 20s (user-verified on LTX Desktop — the code's `get_ltx_frame_count()` cap of 10s only governs timeline planning utilities, not the API payload).
- **Intercut mode**: Default mode that scans vocal audio for silence gaps to create alternating Vocal/Action shots.
- **Z-Image First Frame mode**: Before generating a video, sends the video prompt to LTX's image endpoint to produce a 1920×1080 first-frame image, then passes it as `imagePath` conditioning into the video generation call. First frames are saved to `first_frames/` and never reused.
- **Vocal chain mode** (`vocal_chain_mode` flag in `generate_video_for_shot()`): Chains consecutive Vocal shots for seamless visual continuity. For each Vocal shot N that precedes another Vocal shot N+1, the system generates shot N **one second longer** than its CSV duration, extracts the frame at `original_dur + 1/24` seconds (the first frame past the intended end) and saves it as `first_frames/{shot_id}_chain_out.jpg`. Shot N+1 is conditioned on this look-ahead frame via `imagePath`, avoiding a duplicate first frame at the cut. Assembly trims shot N to its original duration automatically via `subclip(0, snapped_dur)`. If shot N is at its resolution's max duration, `_get_chain_extension_resolution()` downgrades to the next resolution tier (1080p→720p, 720p→540p) so the extension can proceed. The `chain_out.jpg` freshness is validated by mtime comparison before use — if it is older than the predecessor's video, Synesthesia falls back to `extract_last_frame()`.

## LTX Desktop API (http://127.0.0.1:8000/api)

Reverse-engineered from LTX-API-Files/. No official public docs exist.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/generate | Generate a video clip |
| POST | {discovered}/generate/image | Generate a still image (Z-image) — path discovered via OpenAPI |
| POST | /api/generate/cancel | Cancel active generation |
| GET  | /api/generation/progress | Poll generation progress |

### POST /api/generate — GenerateVideoRequest
- `prompt` (str, required)
- `resolution` (str): "512p", "540p", "720p", "1080p" — UI labels only, actual output varies
- `model` (str): "fast" | "pro"
- `cameraMotion` (str): "none" | "dolly_in" | "dolly_out" | "dolly_left" | "dolly_right" | "jib_up" | "jib_down" | "static" | "focus_shift"
- `negativePrompt` (str)
- `duration` (str): seconds as string, e.g. "3"
- `fps` (str): "24"
- `audio` (str): "true" | "false"
- `imagePath` (str | null): absolute path to first-frame image for i2v conditioning
- `audioPath` (str | null): absolute path to audio file for a2v
- `aspectRatio` (str): "16:9" | "9:16"

Response: `{ "status": str, "video_path": str | null }`

### POST {discovered}/generate/image — GenerateImageRequest
Built into LTX Desktop. Uses ZitImageGenerationPipeline (local GPU, ZImage model).
The exact path varies by LTX Desktop version — Synesthesia discovers it at startup by querying
`GET {host}/openapi.json` and finding the route whose requestBody references `GenerateImageRequest`.
- `prompt` (str, required)
- `width` (int): default 1024 — use 1920 for 1080p
- `height` (int): default 1024 — use 1080 for 1080p
- `numSteps` (int): default 4 — inference steps
- `numImages` (int): default 1

Response: `{ "status": str, "image_paths": list[str] | null }` — local absolute paths on the LTX Desktop machine.
Note: The POST blocks until generation completes; `image_paths[0]` in the response is the result.
Discovery: `_discover_zimage_url()` in `video.py` handles this — caches after first successful call.

### GET /api/generation/progress
Response: `{ "status": str, "phase": str, "progress": int, "currentStep": int|null, "totalSteps": int|null }`

### Important notes
- Only one generation can run at a time (video or image)
- Both video and image generation are polled via the same `/generation/progress` endpoint
- `imagePath` and `audioPath` must be absolute paths
- LTX output resolutions are multiples of 32 and vary by audio attachment — do NOT hardcode

## LTX2.3-Multifunctional API (http://127.0.0.1:3000/api)

Community fork of LTX Desktop with LoRA support, batch generation, retaking, and upscaling. Full docs in `API (LTX2.3-Multifunctional-main).md`. Selected via the "LTX2.3-Multifunctional" option in Tab 5.

### Key differences from LTX Desktop

| Concern | LTX Desktop | LTX2.3-Multifunctional |
|---------|------------|------------------------|
| Default port | 8000 | 3000 |
| Auth token | Optional Bearer token | None |
| Cancel endpoint | `POST /api/generate/cancel` | `POST /api/system/cancel` |
| Z-Image endpoint | Discovered via `GET {host}/openapi.json` (path varies) | Fixed: `POST {host}/api/generate-image` |
| LoRA support | No | `loraPath` (str\|null) + `loraStrength` (float) on `/api/generate` |

### ⚠️ Strict string types on POST /api/generate

LTX2.3-Multifunctional validates field types strictly and returns HTTP 422 if they are wrong. These three fields **must be strings**, not native Python types:

| Field | Wrong | Correct |
|-------|-------|---------|
| `duration` | `5` (int) | `"5"` (str) |
| `fps` | `24` (int) | `"24"` (str) |
| `audio` | `False` (bool) | `"false"` (str) |

Synesthesia handles this in `video.py` `worker()`: before `requests.post`, if `config.VIDEO_BACKEND == "LTX2.3-Multifunctional"`, a copy of the payload is made and these three fields are coerced to strings.

### LoRA support

- `GET /api/loras` — returns `{ "loras": [{ "name": str, "path": str }], "loras_dir": str }`. Use `entry["path"]` as the `loraPath` value in generate requests.
- Tab 3 shows a LoRA dropdown (hidden for other backends). The 🔄 button fetches the current list. Selected path is saved as `last_lora` in project settings.
- `loraPath` / `loraStrength` are only injected into the payload when `config.VIDEO_BACKEND == "LTX2.3-Multifunctional"` and the selection is not `"None"`.

### Z-Image with LTX2.3-Multifunctional

`_discover_zimage_url()` in `video.py` short-circuits for this backend — returns `{host}/api/generate-image` directly without OpenAPI discovery. The response format `{ "status": str, "image_paths": list[str] }` is identical to LTX Desktop, so no other changes are needed.

