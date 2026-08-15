import gradio as gr

import config
from models import LLMBridge

_DEFAULT_URLS = {
    "LTX Desktop":            "http://127.0.0.1:8000/api",
    "Wan2GP":                 "http://127.0.0.1:7862/api",
    "LTX2.3-Multifunctional": "http://127.0.0.1:3000/api",
}


def build(pm_state, llm_dropdown):
    """Build Tab 5: Settings."""

    with gr.Tab("5. Settings"):
        gr.Markdown("### ⚙️ Global Settings")
        gr.Markdown("These settings apply globally across all projects and are saved immediately on click.")
        with gr.Row():
            video_backend_drp = gr.Dropdown(
                choices=["LTX Desktop", "Wan2GP", "LTX2.3-Multifunctional"],
                value=config.VIDEO_BACKEND,
                label="Video Generation Backend",
            )
            video_api_url_in = gr.Textbox(
                label="Video Backend API URL",
                value=config.LTX_BASE_URL,
                placeholder="http://127.0.0.1:8000/api",
            )
            ltx_auth_token_in = gr.Textbox(
                label="LTX Desktop Auth Token",
                value=config.LTX_AUTH_TOKEN,
                placeholder="Leave blank for no auth (vanilla LTX Desktop)",
                info="Set to match LTX_AUTH_TOKEN in your LTX Desktop fork. Leave blank for stock LTX Desktop.",
            )
            lm_url_in = gr.Textbox(
                label="LLM API URL (LM Studio / llama.cpp)",
                value=config.LM_STUDIO_URL,
                placeholder="http://127.0.0.1:1234/v1",
            )
            comfyui_url_in = gr.Textbox(
                label="ComfyUI API URL",
                value=config.COMFYUI_URL,
                placeholder="http://127.0.0.1:8188",
                info="Base URL for ComfyUI. Used when Z-Image Backend is set to ComfyUI in Tab 3.",
            )
            electricity_cost_in = gr.Number(
                label="Electricity Cost ($/kWh)",
                value=config.ELECTRICITY_COST,
                minimum=0.0,
                maximum=10.0,
                step=0.001,
                info="Used for render cost estimates. Default: $0.1805 (18.05¢/kWh)",
            )
            system_wattage_in = gr.Number(
                label="System Wattage (W)",
                value=config.SYSTEM_WATTAGE,
                minimum=50,
                maximum=5000,
                step=10,
                info="Full system draw during video generation. Default: 600W (RTX 5090 system)",
            )
        _gpu_choices = config.get_gpu_list()
        _gpu_default_idx = config.GPU_MONITOR_INDEX
        _gpu_default = next((c for c in _gpu_choices if c.startswith(str(_gpu_default_idx))), _gpu_choices[0])
        gpu_monitor_drp = gr.Dropdown(
            label="GPU to Monitor (VRAM leakage detection)",
            choices=_gpu_choices,
            value=_gpu_default,
            info="Select the GPU used by LTX Desktop. Requires pynvml. For multi-GPU systems.",
        )
        with gr.Row():
            save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
            calibration_reset_btn = gr.Button("🔄 Reset Render Calibration", variant="secondary")
        with gr.Row():
            make_default_btn = gr.Button("📌 Make Current Project Settings Default", variant="secondary")
        settings_status = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("📊 Render Calibration Stats", open=False):
            calibration_stats_txt = gr.Textbox(
                label="Current Calibration Data",
                value=config.get_calibration_summary(),
                interactive=False,
                lines=8,
            )
            calibration_refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")

        with gr.Accordion("Advanced: LLM Prompt Templates", open=False):
            gr.Markdown("Customize the prompts sent to the local LLM for each generation step. "
                        "Templates use `{placeholder}` syntax for dynamic values. "
                        "Changes auto-save per project.")
            reset_templates_btn = gr.Button("Reset All Templates to Defaults")

            with gr.Accordion("Plot Generation Template", open=False):
                plot_sys_prompt_in = gr.Textbox(value=config.DEFAULT_PLOT_SYSTEM_PROMPT_MUSIC, label="System Prompt (Music Video Mode)", lines=2)
                plot_user_template_in = gr.Textbox(value=config.DEFAULT_PLOT_USER_TEMPLATE_MUSIC, label="User Prompt Template (Music Video Mode)", lines=4)
                gr.Markdown("*Placeholders: `{concept}`, `{lyrics}`, `{timeline}`*")
                plot_sys_prompt_scripted_in = gr.Textbox(value=config.DEFAULT_PLOT_SYSTEM_PROMPT_SCRIPTED, label="System Prompt (Scripted Mode)", lines=2)
                plot_user_template_scripted_in = gr.Textbox(value=config.DEFAULT_PLOT_USER_TEMPLATE_SCRIPTED, label="User Prompt Template (Scripted Mode)", lines=4)
                gr.Markdown("*Placeholders: `{concept}`, `{timeline}`*")

            with gr.Accordion("Performance Description Template", open=False):
                perf_sys_prompt_in = gr.Textbox(value=config.DEFAULT_PERF_SYSTEM_PROMPT_MUSIC, label="System Prompt (Music Video Mode)", lines=2)
                perf_user_template_in = gr.Textbox(value=config.DEFAULT_PERF_USER_TEMPLATE_MUSIC, label="User Prompt Template (Music Video Mode)", lines=4)
                gr.Markdown("*Placeholders: `{concept}`, `{plot}`, `{gender_instruction}`*")
                perf_sys_prompt_scripted_in = gr.Textbox(value=config.DEFAULT_PERF_SYSTEM_PROMPT_SCRIPTED, label="System Prompt (Scripted Mode)", lines=2)
                perf_user_template_scripted_in = gr.Textbox(value=config.DEFAULT_PERF_USER_TEMPLATE_SCRIPTED, label="User Prompt Template (Scripted Mode)", lines=4)
                gr.Markdown("*Placeholders: `{concept}`, `{plot}`, `{gender_instruction}`*")

            with gr.Accordion("Video Prompt Generation Template (Bulk)", open=False):
                concepts_bulk_template_in = gr.Textbox(value=config.BULK_PROMPT_TEMPLATE, label="Intercut / All Action Template", lines=6)
                gr.Markdown("*Placeholders: `{lyrics}`, `{plot}`, `{shot_list}`*")
                concepts_vocals_template_in = gr.Textbox(value=config.ALL_VOCALS_PROMPT_TEMPLATE, label="All Vocals Template", lines=6)
                gr.Markdown("*Placeholders: `{lyrics}`, `{plot}`, `{performance_desc}`, `{shot_list}`*")
                concepts_scripted_template_in = gr.Textbox(value=config.SCRIPTED_PROMPT_TEMPLATE, label="Scripted Template", lines=6)
                gr.Markdown("*Placeholders: `{gender}`, `{character_desc}`, `{concept}`, `{shot_list}`*")

            with gr.Accordion("Character Bible Template", open=False):
                bible_sys_prompt_in = gr.Textbox(value=config.CHARACTER_BIBLE_SYSTEM_PROMPT, label="System Prompt", lines=2)
                bible_user_template_in = gr.Textbox(value=config.CHARACTER_BIBLE_USER_TEMPLATE, label="User Prompt Template", lines=8)
                gr.Markdown("*Placeholders: `{shot_prompts}`*")

            with gr.Accordion("Single Shot Regeneration Template (Used in Tab 3)", open=False):
                prompt_template_in = gr.Textbox(value=config.DEFAULT_CONCEPT_PROMPT, label="Single Shot Prompt Template", lines=4)
                gr.Markdown("*Placeholders: `{plot}`, `{prev_shot}`, `{start}`, `{duration}`, `{type}`*")

            with gr.Accordion("Z-Image First Frame Prompt Conversion (Used in Tab 3)", open=False):
                zimage_template_in = gr.Textbox(value=config.DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE, label="Z-Image Prompt Conversion Template", lines=3)
                gr.Markdown("*Placeholder: `{prompt}` — the video prompt to be converted to a still-image prompt.*")

        with gr.Accordion("Wan2GP Setup Instructions", open=(config.VIDEO_BACKEND == "Wan2GP")) as wan2gp_accordion:
            gr.Markdown("""
**Wan2GP** is a free open-source video generation app that runs on GPUs with less VRAM than LTX Desktop.
The Wan2.1 1.3B model runs on ~6 GB VRAM.

> **Note:** Wan2GP does not support audio-guided generation. Vocal shots will be generated
> from their text prompt only — no lip-sync. All other Synesthesia features work normally.

---

#### Manual Install

1. Clone Wan2GP: `git clone https://github.com/deepbeepmeep/Wan2GP` and follow its README
2. Install Flask in your Wan2GP virtual environment: `pip install flask`
3. Copy `wan2gp_server.py` from the Synesthesia folder into your Wan2GP folder
4. Start the bridge server:
   ```
   python wan2gp_server.py --model t2v_1.3B
   ```
5. Set **Video Backend API URL** above to `http://127.0.0.1:7862/api` and click **Save Settings**

---

#### Pinokio Install

Pinokio sandboxes Wan2GP in its own Python environment — you must use Pinokio's Python, not the system Python.

1. Find your Pinokio home directory: open **Pinokio → Settings** (shown at the top of the page). Call this path `<pinokio_home>`.
2. Install Flask using **Pinokio's pip** (run once in a terminal):
   ```
   <pinokio_home>\\api\\wan2gp.git\\app\\env\\Scripts\\pip.exe install flask
   ```
3. Copy `wan2gp_server.py` from the Synesthesia folder into:
   `<pinokio_home>\\api\\wan2gp.git\\app\\`
4. **⚠️ Stop Pinokio's Wan2GP UI before starting the bridge** — both try to load the model and will conflict.
5. Start the bridge using **Pinokio's Python**:
   ```
   <pinokio_home>\\api\\wan2gp.git\\app\\env\\Scripts\\python.exe wan2gp_server.py --model t2v_1.3B
   ```
6. Set **Video Backend API URL** above to `http://127.0.0.1:7862/api` and click **Save Settings**

---

**Available models:**

| Model | VRAM | Notes |
|-------|------|-------|
| `t2v_1.3B` | ~6 GB | Good quality, runs on modest GPUs |
| `t2v` | ~20 GB | Better quality (14B model) |
| `ltx2_22B_distilled` | ~24 GB | Best quality (same engine as LTX Desktop) |
""")

        with gr.Accordion("LTX2.3-Multifunctional Setup Instructions",
                           open=(config.VIDEO_BACKEND == "LTX2.3-Multifunctional")) as ltx23_accordion:
            gr.Markdown("""
**LTX2.3-Multifunctional** is a community fork of LTX Desktop that adds LoRA support, batch generation,
video retaking, upscaling, and more.

> **Note:** LTX2.3-Multifunctional does **not** require an auth token — leave the Auth Token field blank.

---

#### Setup

1. Clone or download [LTX2.3-Multifunctional](https://github.com/hero8152/LTX2.3-Multifunctional)
   and follow its README to install dependencies.
2. Start the server:
   ```
   run.bat
   ```
3. The API runs on port **3000** by default. Set **Video Backend API URL** above to
   `http://127.0.0.1:3000/api` and click **Save Settings**.

---

#### LoRA Support

When this backend is active, a **LoRA** dropdown appears in Tab 3. Click **🔄** to load the list of
LoRA files from the backend's configured LoRA directory. Set the LoRA directory from within
LTX2.3-Multifunctional's own settings if needed.
""")

        backend_switch_status = gr.Textbox(label="", interactive=False, visible=True)

    # --- Events ---

    def on_backend_change(backend, pm):
        if backend == "Wan2GP":
            df = pm.load_dataframe()
            if df is not None and not df.empty and "Total_Frames" in df.columns:
                oversized = df[df["Total_Frames"] > 81]
                if not oversized.empty:
                    shot_ids = ", ".join(str(s) for s in oversized["Shot_ID"].tolist()[:5])
                    return (
                        gr.update(value="LTX Desktop"),
                        gr.update(value=_DEFAULT_URLS["LTX Desktop"]),
                        gr.update(open=False),
                        gr.update(open=False),
                        gr.update(value=f"❌ Cannot switch to Wan2GP: shots {shot_ids} exceed 81 frames (3s max). Shorten or regenerate timeline first."),
                    )
            if df is not None and not df.empty and "Video_Path" in df.columns:
                has_videos = df["Video_Path"].notna() & (df["Video_Path"].astype(str).str.strip() != "")
                if has_videos.any():
                    return (
                        gr.update(value="LTX Desktop"),
                        gr.update(value=_DEFAULT_URLS["LTX Desktop"]),
                        gr.update(open=False),
                        gr.update(open=False),
                        gr.update(value="❌ Cannot switch to Wan2GP: LTX-generated videos already exist. Delete them first to avoid timing mismatches."),
                    )
        url = _DEFAULT_URLS.get(backend, config.LTX_BASE_URL)
        is_wan2gp = (backend == "Wan2GP")
        is_ltx23 = (backend == "LTX2.3-Multifunctional")
        return (
            gr.update(),
            gr.update(value=url),
            gr.update(open=is_wan2gp),
            gr.update(open=is_ltx23),
            gr.update(value=""),
        )

    video_backend_drp.change(
        on_backend_change,
        inputs=[video_backend_drp, pm_state],
        outputs=[video_backend_drp, video_api_url_in, wan2gp_accordion, ltx23_accordion, backend_switch_status],
    )

    def handle_save_settings(video_url, ltx_auth_token, lm_url, comfyui_url, backend, electricity_cost, system_wattage, gpu_monitor):
        settings = {
            "ltx_base_url": video_url,
            "ltx_auth_token": ltx_auth_token,
            "lm_studio_url": lm_url,
            "comfyui_url": comfyui_url,
            "video_backend": backend,
            "electricity_cost": electricity_cost,
            "system_wattage": system_wattage,
            "gpu_monitor_index": gpu_monitor,
        }
        status = config.save_global_url_settings(settings)
        return status, gr.update(choices=LLMBridge().get_models())

    save_settings_btn.click(
        handle_save_settings,
        inputs=[video_api_url_in, ltx_auth_token_in, lm_url_in, comfyui_url_in, video_backend_drp, electricity_cost_in, system_wattage_in, gpu_monitor_drp],
        outputs=[settings_status, llm_dropdown],
    )

    calibration_reset_btn.click(
        config.reset_render_calibration,
        outputs=[settings_status],
    ).then(
        config.get_calibration_summary,
        outputs=[calibration_stats_txt],
    )

    calibration_refresh_btn.click(
        config.get_calibration_summary,
        outputs=[calibration_stats_txt],
    )

    def handle_make_default(pm):
        if not pm or not pm.current_project:
            return "❌ No project loaded. Load a project first."
        project_settings = pm.load_project_settings()
        to_save = {k: v for k, v in project_settings.items() if k in config.GLOBALIZABLE_KEYS}
        if not to_save:
            return "⚠️ No globalizable settings found in current project."
        config.save_global_url_settings(to_save)
        return f"✅ {len(to_save)} settings saved as global defaults."

    make_default_btn.click(handle_make_default, inputs=[pm_state], outputs=[settings_status])

    # --- Template auto-save and reset ---

    _template_inputs = [
        plot_sys_prompt_in, plot_user_template_in, plot_sys_prompt_scripted_in, plot_user_template_scripted_in,
        perf_sys_prompt_in, perf_user_template_in, perf_sys_prompt_scripted_in, perf_user_template_scripted_in,
        concepts_bulk_template_in, concepts_vocals_template_in, concepts_scripted_template_in,
        bible_sys_prompt_in, bible_user_template_in,
        prompt_template_in, zimage_template_in,
        pm_state,
    ]

    def auto_save_templates(p_sys_m, p_user_m, p_sys_s, p_user_s,
                            pf_sys_m, pf_user_m, pf_sys_s, pf_user_s,
                            c_bulk, c_vocals, c_scripted,
                            b_sys, b_user, prompt_tmpl, zi_template, pm):
        if pm and pm.current_project:
            pm.save_project_settings({
                "plot_sys_prompt_music": p_sys_m, "plot_user_template_music": p_user_m,
                "plot_sys_prompt_scripted": p_sys_s, "plot_user_template_scripted": p_user_s,
                "perf_sys_prompt_music": pf_sys_m, "perf_user_template_music": pf_user_m,
                "perf_sys_prompt_scripted": pf_sys_s, "perf_user_template_scripted": pf_user_s,
                "concepts_bulk_template": c_bulk, "concepts_vocals_template": c_vocals,
                "concepts_scripted_template": c_scripted,
                "bible_sys_prompt": b_sys, "bible_user_template": b_user,
                "prompt_template": prompt_tmpl,
                "zimage_prompt_template": zi_template,
            })

    for _tmpl_comp in [plot_sys_prompt_in, plot_user_template_in, plot_sys_prompt_scripted_in, plot_user_template_scripted_in,
                       perf_sys_prompt_in, perf_user_template_in, perf_sys_prompt_scripted_in, perf_user_template_scripted_in,
                       concepts_bulk_template_in, concepts_vocals_template_in, concepts_scripted_template_in,
                       bible_sys_prompt_in, bible_user_template_in, prompt_template_in, zimage_template_in]:
        _tmpl_comp.blur(auto_save_templates, inputs=_template_inputs)

    def reset_templates():
        return (
            config.DEFAULT_PLOT_SYSTEM_PROMPT_MUSIC, config.DEFAULT_PLOT_USER_TEMPLATE_MUSIC,
            config.DEFAULT_PLOT_SYSTEM_PROMPT_SCRIPTED, config.DEFAULT_PLOT_USER_TEMPLATE_SCRIPTED,
            config.DEFAULT_PERF_SYSTEM_PROMPT_MUSIC, config.DEFAULT_PERF_USER_TEMPLATE_MUSIC,
            config.DEFAULT_PERF_SYSTEM_PROMPT_SCRIPTED, config.DEFAULT_PERF_USER_TEMPLATE_SCRIPTED,
            config.BULK_PROMPT_TEMPLATE, config.ALL_VOCALS_PROMPT_TEMPLATE, config.SCRIPTED_PROMPT_TEMPLATE,
            config.CHARACTER_BIBLE_SYSTEM_PROMPT, config.CHARACTER_BIBLE_USER_TEMPLATE,
            config.DEFAULT_CONCEPT_PROMPT,
            config.DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE,
        )

    reset_templates_btn.click(reset_templates, outputs=[
        plot_sys_prompt_in, plot_user_template_in,
        plot_sys_prompt_scripted_in, plot_user_template_scripted_in,
        perf_sys_prompt_in, perf_user_template_in,
        perf_sys_prompt_scripted_in, perf_user_template_scripted_in,
        concepts_bulk_template_in, concepts_vocals_template_in, concepts_scripted_template_in,
        bible_sys_prompt_in, bible_user_template_in,
        prompt_template_in,
        zimage_template_in,
    ])

    return {
        "video_backend_drp": video_backend_drp,
        "plot_sys_prompt_in": plot_sys_prompt_in,
        "plot_user_template_in": plot_user_template_in,
        "plot_sys_prompt_scripted_in": plot_sys_prompt_scripted_in,
        "plot_user_template_scripted_in": plot_user_template_scripted_in,
        "perf_sys_prompt_in": perf_sys_prompt_in,
        "perf_user_template_in": perf_user_template_in,
        "perf_sys_prompt_scripted_in": perf_sys_prompt_scripted_in,
        "perf_user_template_scripted_in": perf_user_template_scripted_in,
        "concepts_bulk_template_in": concepts_bulk_template_in,
        "concepts_vocals_template_in": concepts_vocals_template_in,
        "concepts_scripted_template_in": concepts_scripted_template_in,
        "bible_sys_prompt_in": bible_sys_prompt_in,
        "bible_user_template_in": bible_user_template_in,
        "prompt_template_in": prompt_template_in,
        "zimage_template_in": zimage_template_in,
    }
