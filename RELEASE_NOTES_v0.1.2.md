## Synesthesia AI Video Director — v0.1.2

### New Features

- **Vocal Shot Chaining** — tick *Chain consecutive vocal shots* in Tab 3 before batch generation to eliminate duplicate first frames at cut points between consecutive Vocal shots. Instead of passing the last frame of shot N directly as the first frame of shot N+1 (which produces a near-freeze at the cut), Synesthesia generates each chaining shot **one second longer** than its timeline duration, extracts the frame just past the intended end as a *look-ahead chain frame*, and uses that to condition the next shot. Assembly automatically trims the extra second. When a shot is already at its resolution's maximum duration, generation is transparently downgraded one resolution tier to make room for the extension (1080p 5s → 720p, 720p 10s → 540p). Works with both batch generation and *Generate Additional Version*. Stale look-ahead frames (from shots re-rendered without chain mode) are detected automatically via file modification time and discarded in favour of a safe fallback.

- **Per-Shot Prompt Override** — the *Full Prompt Text* accordion in Tab 3 is now editable. Click **💾 Save Override** to lock in your exact text for that shot, bypassing character bible injection, style wrapping, and director credit entirely (style negative prompts still apply). An **⚡** indicator appears when an override is active; **🗑 Clear Override** restores normal assembled behaviour. Overrides auto-clear when the base Video Prompt is edited, on *Regenerate AND Prompt*, and during bulk concept regeneration.

- **Make Current Project Settings Default** (Tab 5) — new button that promotes all current project settings into `global_settings.json` so every new project starts with your preferred prompt templates, timeline defaults, and generation preferences. API endpoints and hardware settings are not affected.

### Improvements

- **Settings persistence — full two-tier save/restore** — all Tab 2 and Tab 3 controls now auto-save to the project's `settings.json` and are fully restored when the project is reloaded. This includes: Z-Image sub-settings and first-frame mode, resolution, style, director, versions per shot, camera motion, vocal prompt mode, generation mode, singer gender, first-frame prompt style/director, and the new vocal chain checkbox. New projects are seeded from the current global defaults on creation.

- **Improved Z-Image prompt conversion template** — the default LLM instruction for converting video prompts into still-image first-frame prompts has been rewritten with clearer, more directive language, resulting in better first-frame compositions out of the box.

- **run-dev.bat** — development launcher no longer opens a duplicate Claude Code window if a session is already running.

### Bug Fixes

- **Tab 4 gallery crash** — `get_project_renders()` no longer appends `(None, caption)` entries when FFmpeg thumbnail generation fails; gallery data and render path lists are now kept in sync, fixing a `ValueError` that could crash Tab 4 when switching to it.
- **`singer_gender` not restoring on project load** — the singer gender selection was silently reset to the default whenever a project was loaded; it now restores correctly from the project's saved settings.
- **`ValueError` on float shot duration in chain code** — shot durations stored as LTX-snapped floats (e.g. `3.0417`) were being parsed with `int()` directly, raising `ValueError: invalid literal for int() with base 10: '3.0417'`. Fixed with `int(float(...))`.

### Prerequisites

- [LM Studio](https://lmstudio.ai/) or any OpenAI-compatible local LLM server
- [LTX Desktop](https://ltx.studio/) or [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
- Python 3.8+
- FFmpeg on system PATH

### Quick Start (Windows)

```
git clone https://github.com/RowanUnderwood/Synesthesia-AI-Video-Director.git
cd Synesthesia-AI-Video-Director
run.bat
```

`run.bat` handles installation automatically on first run. See [README](https://github.com/RowanUnderwood/Synesthesia-AI-Video-Director/blob/main/README.MD) for full setup instructions.
