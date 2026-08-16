# Synesthesia AI Video Director

Synesthesia AI Video Director is an AI-powered music video creation tool. It analyses vocal audio to build a shot timeline, uses a local LLM to generate cinematic video prompts, sends those prompts to a local AI video engine for generation, and assembles the final video with your song audio.

## Prerequisites

You will need two external applications running alongside this tool:

- **[LM Studio](https://lmstudio.ai/)** or **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (`llama-server.exe`) — Local LLM inference server. Load a model (qwen3-8b or similar recommended) and start the server on the default port (`http://127.0.0.1:1234`). LM Studio provides the language model that writes video prompts and plot summaries. If using llama-server, start it with at least **32K context** to handle large shot lists: `llama-server.exe -m your-model.gguf --ctx-size 32768`.

- **Video generation backend** — Choose one of the following:
  - **[LTX Desktop](https://ltx.io/ltx-desktop)** *(recommended)* — Local AI video generation engine. Start the LTX Desktop server on the default port (`http://127.0.0.1:8000`). Requires ~20+ GB VRAM.
  - **[Wan2GP](https://github.com/deepbeepmeep/Wan2GP)** *(low-VRAM alternative)* — Open-source video generation app. The Wan2.1 1.3B model runs on ~6 GB VRAM. Requires running `wan2gp_server.py` as an HTTP bridge (see [Wan2GP Setup](#wan2gp-alternative-backend) below).

> **LTX Desktop VRAM Bypass:** LTX Desktop may refuse to run if your GPU VRAM is below its default threshold. You can bypass this on a **fresh install** (before launching the app for the first time) by editing `runtime_policy.py` located in `LTX Desktop\resources\backend\runtime_config`. Find the Windows VRAM check and lower the threshold to below your available VRAM:
> ```python
> if system == "Windows":
>     if not cuda_available:
>         return True
>     if vram_gb is None:
>         return True
>     return vram_gb < 22
> ```
> Change `22` to a value less than or equal to your GPU's VRAM (e.g. `< 8` for an 8 GB card). Then start the app — it will work, though generation may be slower.

You will also need the following installed on your system:

- **[Python](https://www.python.org/downloads/)** — Version 3.10 or higher. During installation on Windows, make sure to check **"Add Python to PATH"**.
- **[FFmpeg](https://ffmpeg.org/)** — Required by `pydub` for audio processing. Must be installed and available on your system PATH.
- **[Git](https://git-scm.com/)** — Required for cloning the repository and receiving updates. Windows users can install [Git for Windows](https://git-scm.com/download/win).

## Quick Start (Windows)

For Windows users who prefer not to use the command line, a single launcher batch file handles everything:

1. **Clone the repository** — Open a terminal and run:

   ```
   git clone https://github.com/RowanUnderwood/Synesthesia-AI-Video-Director.git
   ```

   Or download and extract the ZIP from GitHub.

2. **Configure `run.bat`** — Open `run.bat` in a text editor and set the paths at the top to match your machine:

   ```bat
   set GIT_BRANCH=main
   set LM_STUDIO_PATH=C:\Users\YourName\AppData\Local\Programs\LM Studio\LM Studio.exe
   set LTX_DESKTOP_PATH=C:\Path\To\LTX Desktop\LTX Desktop.exe
   ```

   - `GIT_BRANCH=main` tracks stable releases (latest tag). Change to `dev` to follow the development branch.
   - Set `LM_STUDIO_PATH` and `LTX_DESKTOP_PATH` to where those apps are installed on your machine.

3. **Run** — Double-click **`run.bat`**. On first launch it automatically creates a virtual environment and installs all Python dependencies, then starts the application. The Gradio interface will open in your browser.

   On subsequent launches, `run.bat` checks for updates and prompts you to apply them before starting.

> **Note:** Make sure Python, FFmpeg, and Git are installed and on your PATH before running the batch file (see [Prerequisites](#prerequisites)).

> **LTX Desktop Auth Fix (latest version):** Recent versions of LTX Desktop enable token-based authentication by default, which blocks Synesthesia's API calls with a 401 Unauthorized error. To disable it, open `ltx2_server.py` inside your LTX Desktop installation (`LTX Desktop\resources\backend\ltx2_server.py`) and change:
> ```python
> auth_token = os.environ.get("LTX_AUTH_TOKEN", "")
> ```
> to:
> ```python
> auth_token = ""
> ```
> Save the file and restart LTX Desktop.

## Manual Installation (All Platforms)

1. **Clone the repository**

   ```bash
   git clone https://github.com/RowanUnderwood/Synesthesia-AI-Video-Director.git
   cd Synesthesia-AI-Video-Director
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** `pydub` requires [FFmpeg](https://ffmpeg.org/) to be installed and available on your system PATH.

4. **Start the external services**

   - Open **LM Studio**, load a model, and start the local server.
   - Open **LTX Desktop** and start the API server (or set up the Wan2GP bridge — see below).

5. **Launch the application**

   ```bash
   python app.py
   ```

   The Gradio interface will open in your browser (typically at `http://127.0.0.1:7860`).

## Usage

### Quick-Start Workflow

1. **Create a Project** (Tab 1) — Give your project a name and click *Create New Project*.
2. **Upload Assets** (Tab 1) — Upload an isolated vocal track (for timeline scanning), optionally upload the full song (for final assembly), and paste your lyrics.
   - **Add Lyric Timestamps** runs Whisper on CPU and locally aligns recognized words to your ground-truth lyric lines. It starts with `small`, rejects incoherent line spans, and retries with `medium.en`/`medium` when coverage is poor. First use of either model downloads it.
   - If you already have timed captions, upload an **SBV, SRT, VTT, or LRC** file and click **Import Timed Captions**. Caption text is matched monotonically to the ground-truth lyrics, including repeated choruses.
   - **Export Templates and Instructions for External LLM** creates a ZIP containing the current shot list, a Character Bible CSV (even when blank), and backend-aware directing instructions with timestamped lyrics.
3. **Build the Timeline** (Tab 2) — Choose a mode, adjust silence detection settings, and click *Scan Vocals & Build Timeline*. The app splits your audio into Vocal and Action shots. Shot durations range from 1–10 seconds; shots over 5 seconds require 720p or lower and will auto-downgrade from 1080p.
4. **Generate Prompts** (Tab 2) — Select your LLM model, write a rough concept, then generate the overarching plot and video prompts with the bulk generation button. Use **Character Bibles** (also on Tab 2) to define recurring characters with a consistent visual description — bibles are automatically injected into prompts at generation time so characters look the same across unrelated shots.
5. **Generate Videos** (Tab 3) — Pick a generation mode, resolution, and optionally enable **Z-Image First Frame** mode for improved character consistency, or **Chain consecutive vocal shots** for seamless frame continuity between consecutive vocal shots (see Tab 6 help for details on both). Click *Start Batch Generation*. Your video backend will render each shot.
6. **Review & Assemble** (Tab 4) — Compare video versions for each shot, pick the best take, and assemble the final video with audio.

### Video Modes

The **Mode** dropdown on Tab 2 controls how the shot timeline is built:

| Mode | Description |
|------|-------------|
| **Intercut** | Default. Scans vocals for silence gaps and creates alternating Vocal and Action shots. Requires a vocals audio file. |
| **All Vocals** | Divides the entire audio duration into Vocal-type shots only. Great for performance-focused music videos. |
| **All Action** | Divides the entire audio duration into Action-type shots only. Great for narrative or visual-only videos. |
| **Scripted** | No audio file needed. Specify a total duration or number of shots. All shots are Action type. Designed for short narrative films. |

### Settings

Synesthesia uses a **two-tier settings model**:

- **Global settings** (`global_settings.json`) — API endpoints, cost/wattage, and any settings you've promoted with the **📌 Make Current Project Settings Default** button. Applied to every new project on creation.
- **Project settings** (`projects/<name>/settings.json`) — all Tab 2 and Tab 3 preferences specific to that project (prompt templates, timeline settings, generation preferences, etc.). Saved automatically as you make changes.

**Tab 5 — Settings** controls:

- **Video Generation Backend** — select **LTX Desktop** (default) or **Wan2GP**. The URL field pre-fills with the appropriate default when you change backends.
- **Video Backend API URL** — base URL for the video generation backend (`http://127.0.0.1:8000/api` for LTX Desktop; `http://127.0.0.1:7862/api` for Wan2GP).
- **LLM API URL** — base URL for the local LLM backend. Supports **LM Studio** (default: `http://127.0.0.1:1234/v1`) and **llama.cpp** `llama-server.exe` (default: `http://127.0.0.1:8080/v1`).
- **📌 Make Current Project Settings Default** — promotes the loaded project's prompt templates, timeline defaults, and generation preferences into `global_settings.json` so that every new project you create starts with those values instead of the built-in defaults.

**Tab 3 generation preferences** (resolution, first-frame mode, style, director, versions, camera motion, vocal prompt mode, and Z-Image sub-settings) are all saved per-project automatically and restored when you reload that project.

### Hotkey

Press `Ctrl+R` in the terminal to restart the application without losing project data.

---

## Wan2GP Alternative Backend

For GPUs with limited VRAM that can't run LTX Desktop, **Wan2GP** is a free open-source alternative. The Wan2.1 1.3B model runs on ~6 GB VRAM.

> **Limitation:** Wan2GP does not support audio-guided generation. Vocal shots will be generated from their text prompt only — no lip-sync. All other features work normally.

### Manual Install

1. Clone Wan2GP: `git clone https://github.com/deepbeepmeep/Wan2GP` and install its dependencies per its README
2. Install Flask in your Wan2GP virtual environment: `pip install flask`
3. Copy `wan2gp_server.py` from the Synesthesia folder into your Wan2GP folder
4. Start the bridge server: `python wan2gp_server.py --model t2v_1.3B`
5. In Synesthesia → Tab 5 Settings: select **Wan2GP** backend, verify the URL is `http://127.0.0.1:7862/api`, and click **Save Settings**

### Pinokio Install

Pinokio sandboxes Wan2GP in its own Python environment. You must use Pinokio's Python — not the system Python — to run the bridge.

1. Find your **Pinokio home directory**: open Pinokio → Settings (the path is shown at the top). Call this `<pinokio_home>`. The Wan2GP app lives at `<pinokio_home>\api\wan2gp.git\app\`.
2. Install Flask using **Pinokio's pip** (run once):
   ```
   <pinokio_home>\api\wan2gp.git\app\env\Scripts\pip.exe install flask
   ```
3. Copy `wan2gp_server.py` from the Synesthesia folder into `<pinokio_home>\api\wan2gp.git\app\`.
4. **⚠️ Stop Pinokio's Wan2GP UI before continuing** — both the UI and the bridge try to load the model and will conflict if run together. The Wan2GP Gradio UI runs on port 7861; the bridge uses port 7862 to avoid this conflict.
5. Start the bridge using **Pinokio's Python**:
   ```
   <pinokio_home>\api\wan2gp.git\app\env\Scripts\python.exe wan2gp_server.py --model t2v_1.3B
   ```
6. In Synesthesia → Tab 5 Settings: select **Wan2GP** backend, set the URL to `http://127.0.0.1:7862/api`, and click **Save Settings**.

### Available Models

| Model | VRAM | Notes |
|-------|------|-------|
| `t2v_1.3B` | ~6 GB | Good quality, runs on modest GPUs |
| `t2v` | ~20 GB | Better quality (14B model) |
| `ltx2_22B_distilled` | ~24 GB | Best quality (same engine as LTX Desktop) |

---

## Tips & Workflow

1. **Vocals file is the backbone** — use a clean isolated vocal track for accurate silence detection. Stems from a vocal remover work well.
2. **Iterate on prompts** — use Export/Import CSV to batch-edit prompts in a spreadsheet before spending time on video generation. In All Vocals, All Action, and Scripted modes, you can also redistribute shot lengths by editing the `Duration` column — all timings are recalculated on import.
3. **Generate multiple versions** — set Versions Per Shot to 2–3 and use the Cutting Room to pick the best take for each shot.
4. **Use Regenerate AND Prompt** on shots you're unhappy with — sometimes a fresh LLM pass produces a much better visual concept.
5. **Chain consecutive vocal shots** — tick this checkbox in Tab 3 before batch-generating to eliminate duplicate frames at the cut points between consecutive vocal shots. Synesthesia generates each shot one second longer than needed, extracts a look-ahead frame just past the intended end, and uses it to condition the next shot's first frame. The extra second is trimmed in assembly automatically. For shots at their resolution's maximum duration (1080p at 5s, 720p at 10s), the generation is automatically downgraded one resolution tier to allow the extension.
6. **Strict vs. Fallback assembly** — use Fallback mode to preview your edit before all shots are done, then switch to Strict for the final render.
7. **Hotkey** — press `Ctrl+R` in the terminal window to restart the application quickly without losing your project data.

### ⚠️ Performance: Avoiding LTX RAM Leakage

LTX Desktop can silently overflow from GPU VRAM into system RAM during a session. Once this happens, generation slows by approximately **10x** — a clip that normally takes 15 seconds can take 2–3 minutes.

- **Start from a fresh reboot** before each generation session — this is the single most effective step
- **Single GPU system:** Close LM Studio before starting video generation to prevent VRAM contention. You can reopen it after the batch is done.
- If generation suddenly slows mid-session, restart LTX Desktop (not just Synesthesia) to reclaim memory

---

## Cloud LLM Prompt Template

If you prefer not to run a local LLM, you can use a cloud-based model (such as Claude, ChatGPT, etc.) to generate your video prompts and character bibles. Here's how:

1. **Export** your shot list from Tab 2 using the **Export CSV** button.
2. **Open** your preferred cloud LLM in a browser.
3. **Attach** the exported shot list file to your message.
4. **Paste** the template below into the message, filling in the bracketed placeholders with your own details.
5. **Send** the message and wait for the LLM to return both the completed shot list and the character bibles CSV.
6. **Import** the completed shot list back into Synesthesia using the **Import CSV** button on Tab 2.
7. **Import** the character bibles using the **Import Bibles CSV** button on Tab 2 — this populates the Character Bibles table and refreshes the Characters column in the shot list.

### Template (copy and paste this into your cloud LLM)

```
Create a music video via AI video prompts for the following song. See the attached shot list with durations and frame counts. We need to tell a coherent story using the shots labeled "action" in the type column. Return TWO files:

1. The shot list CSV with each "Video_Prompt" field filled out.
2. A separate character_bibles.csv file with two columns: "character_name" and "description". Include one row per recurring character who appears in the action shots. Each description should be a single dense paragraph covering the character's age, gender, ethnicity, hair colour and style, clothing, and any other distinguishing visual details — written so that an AI video model can reproduce the character's appearance consistently across multiple unrelated shots. For MiniMax H3, include the lead singer so H3 can generate and reuse their character references. For LTX backends, the lead singer may instead be handled separately through the vocal shot description.

The AI video prompt for the vocal shots should always be very similar to the following as we cut to the live performance. We need to focus on consistency and always being closeup so the lip-sync model has enough pixels to work with.

"Handheld dynamic closeup shot of a [describe lead singer here] Dynamic camera movement with slight handheld shake, shallow depth of field, dramatic lighting, 85mm lens, 24fps, high contrast, band silhouettes, energetic atmosphere, cinematic color grading, [describe color palette here] [name of singer] is careful to enunciate each word to the camera to account for their deaf sister's lip reading."

Follow the LTX prompt guide to create each "action" AI video model prompt:

- Establish the shot. Use cinematography terms that match your preferred film genre. Include aspects like scale or specific category characteristics to further refine the style you're looking for.
- Set the scene. Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.
- Describe the action. Write the core action as a natural sequence, flowing from beginning to end.
- Describe your characters by first name only in the prompt itself - the character bible feature will automatically insert the description.
- Identify camera movement(s). Specify when the view should shift and how. Including how subjects or objects appear after the camera motion gives the model a better idea of how to finish the motion.
- Keep your prompt in a single flowing paragraph to give the model a cohesive scene to work with.
- Use present tense verbs to describe movement and action.
- Match your detail to the shot scale. Closeups need more precise detail than wide shots.
- When describing camera movement, focus on the camera's relationship to the subject.
- You should expect to write 4 to 8 descriptive sentences to cover all the key aspects of the prompt.

Lead Singer's gender: [insert gender description here]
Story Idea: [insert story idea here]
Genre: [insert genre tags here]
Lyrics: [insert lyrics here]
```

## License

See [LICENSE](LICENSE) for details.
