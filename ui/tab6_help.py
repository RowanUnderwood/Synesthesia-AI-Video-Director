import gradio as gr


def build():
    """Build Tab 6: Help (static content)."""

    with gr.Tab("6. Help"):
        gr.HTML("""
        <a href="https://www.buymeacoffee.com/jacobpederson" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 181px !important;" >
        </a>
        """)
        gr.Markdown("""
# Synesthesia AI Video Director — User Guide

This application helps you create AI-generated music videos by combining audio analysis, LLM-generated video prompts, and the LTX Desktop video generation engine.

---

## Tab 1 · Project & Assets

**Create a project** by typing a name and clicking *Create New Project*. This sets up all the necessary folders for your project. **Load an existing project** from the dropdown and click *Load Selected Project* — all your previous settings, prompts, and video paths will be restored automatically.

Upload your **vocals audio** (an isolated vocal track — stems work best). The vocals file is used for two things: scanning silence to build the shot timeline, and providing audio sync for generated vocal shots. Optionally upload a **full song** file, which is used as the audio track in the final assembled video.

Paste your **lyrics** in the text box. These are saved with the project and handed to the LLM when generating the overarching plot.

After uploading isolated vocals and pasting ground-truth lyrics, click **Add Lyric Timestamps** to run CPU-only speech recognition and align local word groups back to the original lyric lines. It starts with Whisper `small`, retries with `medium.en` or multilingual `medium` when coverage is poor, and rejects lines with implausible duration or gaps. Confident matches receive `[MM:SS.mmm --> MM:SS.mmm]` prefixes; unmatched lines remain unchanged. Replacing the vocals or editing the lyrics makes the saved alignment stale until you scan again.

When authoritative captions already exist, upload an **SBV, SRT, VTT, or LRC** file and click **Import Timed Captions**. Synesthesia matches caption text to the ground-truth lyric lines in order, so repeated choruses consume repeated caption cues correctly. Caption import avoids Whisper and is the preferred path when good timings are available.

Click **Export Templates and Instructions for External LLM** after building the timeline and timestamping at least one lyric line. The downloaded ZIP contains `shot_list.csv`, `character_bibles.csv` (including a header-only template when no characters exist), and backend-aware instructions with the timestamped lyrics.

---

## Tab 2 · Storyboard

### Step 1 — Build the Timeline

Choose a **Mode** from the dropdown to control how the shot timeline is constructed:

| Mode | Description |
|------|-------------|
| **Intercut** (default) | Scans the vocals audio for silence gaps and creates alternating **Vocal** shots (singing detected) and **Action** shots (silent gaps). Requires a vocals audio file. The silence-detection sliders are only active in this mode. |
| **All Vocals** | Divides the entire audio duration into **Vocal**-type shots only. No silence detection is performed. Ideal for performance-focused music videos where every shot features the singer/band. |
| **All Action** | Divides the entire audio duration into **Action**-type shots only. No silence detection is performed. Ideal for narrative or visual-only videos that don't require lip-sync. |
| **Scripted** | No audio file is needed. You specify a **Total Duration** or **Number of Shots** instead. All shots are Action type. UI labels change from "Singer" to "Main Character", making this mode suited for short narrative films without music. |

Click *Scan Vocals & Build Timeline* (or *Build Timeline* in non-Intercut modes) to generate the shot list.

Adjust the sliders to fine-tune detection (Intercut mode) and shot lengths:
- **Min Silence (ms)** — how long a pause must be to count as silence (Intercut only)
- **Silence Threshold (dB)** — how quiet audio must be to be treated as silent (Intercut only)
- **Shot Duration Mode** — *Fixed* uses the Min Duration for every shot; *Random* picks a random length between Min and Max
- **Min/Max Duration** — the allowed range for shot lengths (1–10 seconds). Shots over 5 seconds require 720p or lower — 1080p selections will automatically downgrade to 720p for those shots.

All shot durations are automatically locked to LTX-compatible frame counts (1–10 second increments at 24 fps).

### Step 2 — Generate Prompts

1. Select your **LLM model** from the dropdown. Click 🔄 to refresh the list from your LLM backend (LM Studio or llama-server).
2. Write a **rough concept** describing the vibe, setting, or mood of the video.
3. Click *Generate Singer, Band & Venue Desc* to create a concise visual description of your performer(s). This is also used as the video prompt for all Vocal shots.
4. Click *Generate Overarching Plot* to produce a cohesive linear narrative based on your concept and lyrics.
5. Click *Generate Video Prompts (Bulk Generation)* to send your entire timeline context, lyrics, and plots over to the LLM. It will return fully conceptualized, sequenced shot descriptions across all rows at once.

**Advanced — Prompt Templates:** Expand this section to customise the fallback instruction sent to the LLM for each Action shot (this is utilized mainly when regenerating single shots in Tab 3). The following placeholders are filled in automatically: `{plot}`, `{prev_shot}`, `{start}`, `{duration}`, `{type}`.

**Data Management:**
- *Export CSV* — download the full shot list with all prompts for external editing
- *Import CSV* — upload an edited CSV to push updates back in. Behaviour depends on the active mode:
  - **Intercut:** Shot IDs and row count must match exactly. Only `Type` and `Video_Prompt` are updated.
  - **All Vocals / All Action:** Edit the `Duration` column to redistribute shot lengths. You may add or remove shots. Total duration must remain close to the original audio length. `Type`, `Video_Prompt`, `Lyrics`, and `Characters` columns are accepted. All timing columns (`Start_Time`, `End_Time`, etc.) are recalculated automatically — do not edit them. Durations are snapped to LTX-compatible values (1–10s whole-second increments) automatically.
  - **Scripted:** Same as All Vocals / All Action, but total duration is not validated (Scripted mode provides its own audio). Note: shots are processed in spreadsheet row order and renumbered S001, S002, … on import.
- *Download Story (.txt)* — export every shot's prompt as a readable text file; includes a Character Bibles section at the end if bibles exist
- *Export Bibles CSV* — download the character bible definitions (`character_name`, `description`) for external editing or cloud LLM use
- *Import Bibles CSV* — upload an edited bibles CSV to replace the current character definitions; the Characters column in the shot table refreshes automatically

---

## Tab 3 · Video Generation

### Batch Generation

Select a **Generation Mode**:
- *Generate Remaining Shots* — only shots that don't yet have a video
- *Generate all Action Shots* / *Generate all Vocal Shots* — target one shot type (skips shots already at version count)
- *Generate Remaining Action Shots* / *Generate Remaining Vocal Shots* — only missing/under-versioned shots of that type
- *Regenerate all Shots* — delete all existing videos and regenerate from scratch

Set how many **Versions per Shot** to generate (1–5). Having multiple versions gives you options to compare in Tab 4. Choose your **Resolution** (540p → 1080p). Shots over 5 seconds will automatically downgrade from 1080p to 720p. Choose a **Style** to apply a prompt style preset to all generated clips (styles can be edited per-project in the Style Editor accordion below the dropdowns). Choose a **Director** to append a directorial style cue to every prompt.

Set **First Frame Mode**:
- *LTX-Native* — standard text-to-video generation
- *Z-Image First Frame* — generates a still image first and uses it as the opening frame for each clip; improves character consistency at the cost of dynamic camera motion (see [Z-Image First Frame Mode](#z-image-first-frame-mode) below)

All of these preferences — mode, versions, resolution, style, director, first-frame mode, Z-Image sub-settings, vocal prompt mode, and camera motion — are **saved per-project automatically** and restored when you reload the project.

Click *Start Batch Generation* to begin. Use **⏸ Pause Queue** to pause between shots, **✖ Cancel All** to clear the remaining queue, or **⏹ Stop** to halt after the current shot finishes.

**Vocal Shot Prompt Mode** controls which prompt drives video generation for Vocal shots:
- *Use Singer/Band Description* — uses the performer/venue description from Tab 2
- *Use Storyboard Prompt* — uses the individually generated shot prompt

### Single Shot Generation

Select a specific shot from the dropdown, optionally edit its prompt inline (changes save automatically), then click *Generate Additional Version* to add another version without deleting existing ones. Use the **Camera Motion** dropdown to apply a specific camera move (dolly, jib, handheld, etc.) to this shot only.

**Generate Additional Version fully supports vocal chain mode** — if the chain checkbox is ticked, it uses the same look-ahead conditioning logic as batch generation.

### Vocal Shot Chaining

**Chain consecutive vocal shots** is a checkbox in Tab 3 that links consecutive Vocal shots together for seamless visual continuity. When enabled, each Vocal shot's generation is conditioned on a *look-ahead frame* taken from the preceding shot, so the cut between them flows as natural motion rather than a jump.

#### How it works

LTX Desktop conditions the first frame of a clip on an `imagePath` input. If the last frame of shot N is passed directly as the first frame of shot N+1, the result is a near-duplicate frame at the cut — a brief visual freeze at 24fps. The chaining system avoids this by:

1. Generating shot N **one second longer** than its timeline duration (a 4s shot becomes 5s)
2. Extracting the frame **one frame past** the intended end of the shot as a *look-ahead chain frame*, saved to `first_frames/{shot_id}_chain_out.jpg`
3. Conditioning shot N+1 on that frame — so N+1 starts from where N *would have continued*, not where it stopped
4. Assembly automatically trims shot N back to its original duration — the extra second is never included in the final video

#### Resolution downgrade for max-duration shots

LTX Desktop has per-resolution duration limits. When a shot is already at its limit, the extra second can't be generated at the same resolution. Synesthesia automatically downgrades to the next tier:

| Original resolution | Shot duration | Extended at |
|---|---|---|
| 1080p | 5s (max) | 720p → 6s |
| 720p | 10s (max) | 540p → 11s |
| 540p | 20s (max) | No downgrade possible — falls back to old method |

When a downgrade occurs, you'll see a **⬇️ Downgrading to…** status message. The shot's timeline duration in the CSV is unchanged; only the generation payload is affected. Assembly resizes all clips to match the project's target resolution, so the downgraded clip is scaled up automatically.

> **Note:** The extended clip is generated at the lower resolution — the first 5 seconds (the actual shot) will be at 720p rather than 1080p when a 1080p 5s shot is chain-extended. If pixel-perfect 1080p is critical for that shot, disable chaining or render it separately without chain mode.

#### Stale chain frame warning

If you **re-render shot N** (the preceding shot) after its successor N+1 is already rendered, shot N gets a new look-ahead frame. Synesthesia will display:

> ⚠️ Shot V002 is already rendered — re-render it to apply the new chain frame.

This means N+1's first frame was conditioned on the *old* version of shot N. Re-rendering N+1 with chain mode will pick up the new chain frame automatically.

The chain frame staleness is also detected automatically: if `chain_out.jpg` is older than the predecessor's video file (i.e. the shot was re-rendered without chain mode), Synesthesia falls back to the old extract-last-frame method rather than using the stale look-ahead.

#### Workflow tips

- Render shots in **timeline order** for the cleanest results — each shot generates its look-ahead frame for the next
- The **first shot in any chain** does not receive a chain frame (there is no predecessor), but it still generates one for the shot that follows
- If chain mode is off when you render a shot, its `chain_out.jpg` is not updated — the next time you render its successor with chain mode on, the stale check will catch this and fall back gracefully
- Works with both **batch generation** and **Generate Additional Version**

### Full Prompt Text

The **Full Prompt Text (character bibles + style + director injected)** accordion shows the exact prompt that will be sent to the video generation backend — the base Video Prompt with character bible descriptions injected at the first occurrence of each character name, wrapped in the selected style template, and the director credit appended.

This field is **editable**. To override the assembled prompt for a specific shot:
1. Edit the text directly in the field (useful for close-ups where you want to shorten bible descriptions, or any other per-shot tweak)
2. Click **💾 Save Override** — the shot will now render with your text verbatim, bypassing character bible injection, style wrapping, and director credit
3. To restore the normal assembled behaviour, click **🗑 Clear Override**

**When an override is active:** the ⚡ indicator appears and the Clear Override button is shown. Style negative prompts still apply during generation (they suppress artefacts independently of the positive prompt).

**Overrides are automatically cleared** when you edit the base Video Prompt for that shot or use *Regenerate AND Prompt*, since the assembled version would change and the override would be stale. Bulk concept regeneration also clears overrides for regenerated shots.

### Gallery & Controls

All generated videos appear in the gallery with their Shot ID and frame count. Click a thumbnail to view it full-size on the right panel. From there you can:
- **🗑️ Delete This Video** — permanently removes the selected video file
- **♻️ Regenerate Video (Same Prompt)** — deletes the selected video and generates a new one with the same prompt
- **✨ Regenerate Video AND Prompt** — generates a fresh LLM prompt first, then generates a new video

---

## Tab 4 · Assembly & Cutting Room

### Version Comparison

Select a shot from the dropdown to see all its generated versions side by side (up to 5 at once).
- Click **⭐ Set as Active** on the version you want to use in the final edit
- Click **✂️ Move to Cutting Room Floor** to move an unwanted version out of the videos folder (it goes to the `cutting_room/` subfolder, not deleted)
- Use **⬅️ Previous** / **➡️ Next** to step through all shots, or **⏮ Prev Multi-Version** / **⏭ Next Multi-Version** to jump only to shots that have more than one version — useful for quickly working through all the choices you need to make.

The tab automatically refreshes its shot list when you switch to it. Use the **Style Filter** dropdown to narrow the comparison view to clips generated with a specific style.

### Final Assembly

Once you're satisfied with your active video selections:
- **🔢 Assemble with Shot Numbers** — assembles the video with each clip's Shot ID burned into the frame. Use this for QA review to identify exactly which clip is which before committing to a final edit.
- **Assemble videos with black fallback** — the main assembly button. Substitutes a black frame for any shot that doesn't have a video yet, so you can preview a partial edit at any stage.

The assembled video is written to the project's `renders/` folder. The full song audio (from Tab 1) is attached if available; otherwise the vocals file is used as a fallback.

### Cutting Room Floor Compilation

**🗂️ Assemble Cutting Room Floor** compiles every version of every shot — both active videos and anything moved to the cutting room — into a single long video, grouped by shot in timeline order with oldest version first. Useful for reviewing all your takes in one pass. The **Audio Mode** dropdown controls audio attachment: *Attach Full Song (Once)* adds the song audio at the start of the compilation, *Loop Full Song* loops it across the full duration, or *Use LTX Clip Audio* keeps the original audio from each individual clip.

### Previous Renders

The bottom of Tab 4 shows a gallery of all assembled videos in the project's `renders/` folder. Click a thumbnail or use the dropdown to load any previous render into the playback player.

---

## Tab 5 · Settings

### Global API Settings

Configure the endpoints used by the application:
- **Video Generation Backend** — select **LTX Desktop** (default) or **Wan2GP** (see below). The URL field pre-fills with the default for the chosen backend.
- **Video Backend API URL** — the base URL for the video generation backend (LTX Desktop default: `http://127.0.0.1:8000/api`; Wan2GP default: `http://127.0.0.1:7862/api`)
- **LLM API URL** — the base URL for the local LLM backend. Supports **LM Studio** (default: `http://127.0.0.1:1234/v1`) and **llama.cpp** `llama-server.exe` (default: `http://127.0.0.1:8080/v1`). When using llama-server, start it with at least **32K context** (`--ctx-size 32768`) for projects with large shot lists.

Click *Save Settings* to apply immediately and refresh the model list. These API settings are stored globally in `global_settings.json` and apply to all projects.

### Two-Tier Settings Model

Synesthesia separates settings into two layers:

| Layer | File | What it holds |
|-------|------|---------------|
| **Global defaults** | `global_settings.json` | API URLs, cost/wattage, and any settings promoted with the button below |
| **Project settings** | `projects/<name>/settings.json` | All Tab 2 content and template customisations, all Tab 3 generation preferences — saved automatically as you work |

When a **new project is created**, its settings file is pre-populated from the current global defaults, so it inherits your preferred prompt templates and generation preferences automatically.

When a **project is loaded**, all Tab 2 and Tab 3 controls are restored to exactly how you left them — including Z-Image settings, resolution, style, director, first-frame mode, versions per shot, and camera motion.

### 📌 Make Current Project Settings Default

This button promotes the loaded project's settings into `global_settings.json` so every new project starts with those values.

**What gets promoted:**
- All LLM prompt templates (plot, performance, concepts bulk, character bible, Z-image conversion)
- Timeline defaults (silence detection, shot duration range, video mode)
- Tab 3 generation preferences (first-frame mode, Z-Image sub-settings, vocal prompt mode, resolution, versions, camera motion, director, style)

**What is never promoted** (these are project-specific by nature): story concept, plot, performance description, singer gender, scripted duration, style overrides, character bibles, Z-image cached prompts/images.

Clicking the button does not overwrite your API URL, wattage, or backend settings.

---

## Wan2GP Alternative Backend

For GPUs with limited VRAM that can't run LTX Desktop, **Wan2GP** is a free open-source alternative. The Wan2.1 1.3B model runs on ~6–8 GB VRAM.

**Limitation:** Wan2GP does not support audio-guided generation. Vocal shots will be generated from their text prompt only (no lip-sync). All other features work normally.

#### Manual Install

1. Clone Wan2GP: `git clone https://github.com/deepbeepmeep/Wan2GP` and install its dependencies per its README
2. Install Flask in your Wan2GP virtual environment: `pip install flask`
3. Copy `wan2gp_server.py` from the Synesthesia folder into your Wan2GP folder
4. Start the bridge server: `python wan2gp_server.py --model t2v_1.3B`
5. In Synesthesia → Tab 5 Settings: select **Wan2GP** backend, verify the URL is `http://127.0.0.1:7862/api`, and click **Save Settings**

#### Pinokio Install

Pinokio sandboxes Wan2GP in its own Python environment. You must use Pinokio's Python — not the system Python — to run the bridge.

1. Find your **Pinokio home directory**: open Pinokio → Settings (the path is shown at the top). Call this `<pinokio_home>`. The Wan2GP app lives at `<pinokio_home>\api\wan2gp.git\app\`.
2. Install Flask using **Pinokio's pip** (run once):
   ```
   <pinokio_home>\api\wan2gp.git\app\env\Scripts\pip.exe install flask
   ```
3. Copy `wan2gp_server.py` from the Synesthesia folder into `<pinokio_home>\api\wan2gp.git\app\`.
4. **⚠️ Stop Pinokio's Wan2GP UI before continuing** — both the UI and the bridge try to load the model and will conflict if run together.
5. Start the bridge using **Pinokio's Python**:
   ```
   <pinokio_home>\api\wan2gp.git\app\env\Scripts\python.exe wan2gp_server.py --model t2v_1.3B
   ```
6. In Synesthesia → Tab 5 Settings: select **Wan2GP** backend, set the URL to `http://127.0.0.1:7862/api`, and click **Save Settings**.

**Available models:**

| Model | VRAM | Quality |
|-------|------|---------|
| `t2v_1.3B` | ~6 GB | Good |
| `t2v` | ~20 GB | Better (14B model) |
| `ltx2_22B_distilled` | ~24 GB | Best (same engine as LTX Desktop) |

---

## Tips & Workflow

1. **Vocals file is the backbone** — use a clean isolated vocal track for accurate silence detection. Stems from a vocal remover work well.
2. **Iterate on prompts** — use Export/Import CSV to batch-edit prompts in a spreadsheet before spending time on video generation.
3. **Generate multiple versions** — set Versions Per Shot to 2–3 and use the Cutting Room to pick the best take for each shot.
4. **Use Regenerate AND Prompt** on shots you're unhappy with — sometimes a fresh LLM pass produces a much better visual concept.
5. **Strict vs. Fallback assembly** — use Fallback mode to preview your edit before all shots are done, then switch to Strict for the final render.
6. **Hotkey** — press `Ctrl+R` in the terminal window to restart the application quickly without losing your project data.

---

## Z-Image First Frame Mode

When enabled in Tab 3, **Z-Image First Frame** mode generates a still image from the video prompt before each clip is rendered. That image is passed to LTX Desktop as the opening frame (image-to-video conditioning), giving each shot a more intentional, controlled starting composition.

**How to use:**
- On Tab 3, set the **First Frame Mode** radio to **Z-Image First Frame**
- No configuration or LTX Desktop modifications are required — the image generation endpoint is built into LTX Desktop
- Generated first-frame images are saved to `{project}/first_frames/`

> Z-Image First Frame only works with the LTX Desktop backend. It has no effect with Wan2GP.

### Trade-off: Character Consistency vs. Camera Motion

Activating Z-Image mode produces a **large dampening effect on camera motion**. Because LTX Desktop must animate from a locked still image, dynamic camera movements and shot variety are significantly reduced — the model tends to produce more static or gently drifting shots rather than the expressive handheld, dolly, or tracking moves it would otherwise generate from a text prompt alone.

**The trade-off:** better character consistency at the cost of dynamic camera work.

To counter-act this, you need to write more assertive, explicit camera motion language in your prompts. The example below demonstrates the level of instruction needed — note the specific handheld tracking description, the subject movement, and the environmental motion (wind) all working together to force dynamism despite the conditioned first frame:

```
(Extreme close-up:1.3) A macro extreme close-up of Aoife, a young Irish woman in her late twenties
with long wavy auburn hair, pale freckled skin, and striking green eyes, wearing a fitted dark green
velvet dress with Celtic knotwork embroidery at the neckline. Dynamic handheld camera tracks Aoife
swaying left and right as she sings into a tiny headset mic. Wind blows her hair gently. Shallow
depth of field, dramatic lighting, 24fps, 85mm lens, high contrast, cinematic color grading, muted
moss greens and warm amber tones with deep shadow blacks and flickers of firelight. Aoife is careful
to enunciate each word to the camera to account for their deaf sister's lip reading.
```

**Key techniques in this example:**
- Use a weighted emphasis token like `(Extreme close-up:1.3)` at the very start to assert shot scale before the model reads anything else
- Name the subject immediately and pair the shot scale with a macro lens description
- Explicitly say *"Dynamic handheld camera tracks [subject] [movement]"* — passive descriptions of camera style are not enough in Z-Image mode; the instruction must be active and specific
- Add independent environmental motion (wind, flickering light) to give the model additional cues for movement that aren't dependent on camera work alone
- End with a behavioural action for the subject to sell the performance feel

---

## LTX Desktop VRAM Bypass

LTX Desktop may refuse to run if your GPU VRAM is below its default threshold. You can bypass this on a **fresh install** (before launching the app for the first time):

1. Navigate to `LTX Desktop\\resources\\backend\\runtime_config\\`
2. Open `runtime_policy.py` in a text editor
3. Find the Windows VRAM check and lower the threshold to below your available VRAM:

```python
if system == "Windows":
    if not cuda_available:
        return True
    if vram_gb is None:
        return True
    return vram_gb < 22
```

4. Change `22` to a value less than or equal to your GPU's VRAM (e.g. `< 8` for an 8 GB card)
5. Save the file, then start LTX Desktop — it will work, though generation may be slower

---

## LTX Desktop Auth Fix (latest version)

Recent versions of LTX Desktop enable token-based authentication by default, which blocks Synesthesia's API calls with a **401 Unauthorized** error. To disable it:

1. Open `ltx2_server.py` inside your LTX Desktop installation:
   `LTX Desktop\resources\backend\ltx2_server.py`
2. Find this line:
```python
auth_token = os.environ.get("LTX_AUTH_TOKEN", "")
```
3. Change it to:
```python
auth_token = ""
```
4. Save the file and restart LTX Desktop.

---

## ⚠️ Performance: Avoiding LTX RAM Leakage

LTX Desktop can silently overflow from GPU VRAM into system RAM during a session. Once this happens, generation slows by approximately **10x** — a clip that normally takes 15 seconds can take 2–3 minutes.

**How to avoid it:**
- **Start from a fresh reboot** before each generation session — this is the single most effective step
- **Single GPU system:** Close LM Studio before starting video generation to prevent VRAM contention. You can reopen it after the batch is done.
- If generation suddenly slows mid-session, restart LTX Desktop (not just Synesthesia) to reclaim memory

**Signs of leakage:** Windows Task Manager shows LTX Desktop processes consuming large amounts of System RAM.

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
Create a music video via AI video prompts for the following song (see the bottom of this text for lyrics). See the attached shot list with durations and frame counts. We need to tell a coherent story using the shots labeled "action" in the type column.  In the vocal type rows, we cut to the singer and band.

Return TWO files:

1. The shot list CSV with each "Video_Prompt" field filled out.

2. A separate character_bibles.csv file with two columns: "character_name" and "description". Include one row per recurring character who appears in the action shots. Each description should be a single dense paragraph covering the character's age, gender, ethnicity, hair color and style, clothing, and any other distinguishing visual details — written so that an AI video model (LTX 2.3) can reproduce the character's appearance consistently across multiple unrelated shots. In the video prompt field for action shots, the characters name will automatically be followed by thier description from the character bible, so there is no need to describe the characters outside of their names.  Avoid using the same name more than once per video prompt.

3. Lead-singer handling depends on the selected backend. For **MiniMax H3**, include the lead singer in the character bibles because H3 generates face and full-body identity references from that entry. For LTX backends, the singer may remain in the separate vocal performance description.

The AI video prompt for the vocal shots should always be very similar to the following as we cut to the live performance. We need to focus on consistency and always being closeup so the lip-sync model has enough pixels to work with.

"Handheld extreme closeup shot of a [describe lead singer here] Dynamic camera movement with slight handheld shake, shallow depth of field, dramatic lighting, 85mm lens, 24fps, high contrast, band silhouettes, energetic atmosphere, cinematic color grading, [describe color palette here] [invent name of singer] is careful to enunciate each word to the camera to account for their deaf sister's lip reading."

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

Lead Singer's gender: 
Story concept: 

Song genre tags:
  
Title:

Lyrics:
```
        """)
