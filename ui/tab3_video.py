import os
import glob
import hashlib
import re
import time
import threading

import gradio as gr
import pandas as pd

import config
from models import LLMBridge, sync_video_directory
from utils import format_eta
from video import (get_project_videos, delete_video_file, generate_video_for_shot,
                   get_video_count_for_shot, apply_character_bibles, resolve_style_data,
                   convert_prompt_for_zimage, generate_comfyui_zimage_first_frame,
                   _will_use_chain_conditioning, _has_valid_cached_first_frame)
from h3 import H3_ASPECT_PRESETS, H3_QUALITY_PRESETS, h3_aspect_cache_key
from render_pipeline import H3RenderPipeline


def build(pm_state, current_proj_var, shared_shot_state):
    """Build Tab 3: Video Generation. Returns dict of exported components."""

    with gr.Tab("3. Video Generation") as tab3_ui:
        selected_vid_path = gr.State("")
        gallery_paths_state = gr.State([])
        backend_state = gr.State(config.VIDEO_BACKEND)

        with gr.Row():
            vid_gen_mode_dropdown = gr.Dropdown(choices=["Generate Remaining Shots", "Regenerate all Shots", "Generate all Action Shots", "Generate all Vocal Shots", "Generate Remaining Action Shots", "Generate Remaining Vocal Shots"], value="Generate Remaining Shots", label="Generation Mode")
            vid_versions_dropdown = gr.Dropdown(choices=[1, 2, 3, 4, 5], value=1, label="Versions per Shot")
            vid_resolution_dropdown = gr.Dropdown(choices=["540p", "720p", "1080p"], value="1080p", label="Resolution")
            vid_firstframe_mode = gr.Radio(
                choices=(["Krea 2 First Frame", "Z-Image First Frame"]
                         if config.VIDEO_BACKEND == "MiniMax H3"
                         else ["LTX-Native", "Z-Image First Frame"]),
                value=("Krea 2 First Frame" if config.VIDEO_BACKEND == "MiniMax H3" else "LTX-Native"),
                label="First Frame Mode"
            )
            vid_zimage_backend = gr.Dropdown(
                choices=(["ComfyUI"] if config.VIDEO_BACKEND == "MiniMax H3"
                         else ["LTX Desktop", "ComfyUI", "ComfyUI-run-ahead"]),
                value=("ComfyUI" if config.VIDEO_BACKEND == "MiniMax H3" else "LTX Desktop"),
                label="Z-Image Backend",
                visible=False,
                info="LTX Desktop: use LTX's built-in Z-Image. ComfyUI: generate via ComfyUI sequentially. ComfyUI-run-ahead: generate next shot's frame on ComfyUI in parallel while LTX renders current shot (dual-GPU).",
            )
            vid_vocal_chain_checkbox = gr.Checkbox(
                value=False,
                label="Chain consecutive vocal shots"
            )
            vid_vocal_prompt_mode = gr.Dropdown(choices=["Use Singer/Band Description", "Use Storyboard Prompt"], value="Use Singer/Band Description", label="Vocal Shot Prompt Mode")

        with gr.Row(visible=(config.VIDEO_BACKEND == "LTX2.3-Multifunctional")) as lora_row:
            vid_lora_dropdown = gr.Dropdown(
                choices=["None"],
                value="None",
                label="LoRA",
                info="LoRA model to apply during generation. Only available with LTX2.3-Multifunctional backend. Click 🔄 to load available LoRAs.",
                scale=4,
            )
            vid_lora_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

        with gr.Row(visible=(config.VIDEO_BACKEND == "MiniMax H3")) as h3_row:
            h3_aspect_dropdown = gr.Dropdown(
                choices=H3_ASPECT_PRESETS, value="3:4 - Photo", label="H3 Aspect Ratio"
            )
            h3_quality_dropdown = gr.Dropdown(
                choices=H3_QUALITY_PRESETS, value="0.65 MP - Balanced", label="H3 Quality"
            )
            h3_lipsync_output_dropdown = gr.Dropdown(
                choices=["RTX Upscaled", "Native"], value="RTX Upscaled", label="H3 Lip-Sync Output"
            )
            h3_custom_width = gr.Number(value=16, precision=0, minimum=1, label="H3 Custom Aspect Width", visible=False)
            h3_custom_height = gr.Number(value=9, precision=0, minimum=1, label="H3 Custom Aspect Height", visible=False)

        with gr.Row():
            vid_style_dropdown = gr.Dropdown(choices=config.STYLE_NAMES, value="None", label="Style")
            vid_director_dropdown = gr.Dropdown(choices=config.DIRECTORS, value="None", label="Directed by")
            vid_custom_director_txt = gr.Textbox(
                label="Custom Director Name", placeholder="Enter director name...",
                visible=False, interactive=True, scale=2
            )

        llm_image_prompt_dropdown = gr.Dropdown(
            choices=["Use video prompt as-is", "Convert with LLM"],
            value="Use video prompt as-is",
            label="First Frame Prompt Mode",
            visible=False,
            info="When 'Convert with LLM' is selected, the video prompt is rewritten as a still-image first-frame prompt before Krea or Z-Image generation. The result is cached to the CSV for reuse."
        )
        first_frame_reuse_dropdown = gr.Dropdown(
            choices=["Use cached prompt", "Use cached image", "Regenerate both on each render"],
            value="Use cached prompt",
            label="First Frame Caching Mode",
            visible=False,
            info=(
                "'Use cached prompt' — reuses the LLM-converted prompt but generates a fresh image each render. "
                "'Use cached image' — reuses both the cached prompt and the generated image file. "
                "'Regenerate both on each render' — bypasses the first-frame caches; always re-runs LLM "
                "conversion and image generation without saving results back to CSV. H3 video-prompt rewrite "
                "caching is controlled separately below."
            )
        )
        with gr.Row(visible=(config.VIDEO_BACKEND == "MiniMax H3")) as h3_prompt_cache_row:
            h3_prompt_cache_dropdown = gr.Dropdown(
                choices=["Reuse cached H3 prompts", "Rewrite H3 prompts each render"],
                value="Reuse cached H3 prompts",
                label="H3 Prompt Rewrite Cache",
                info=(
                    "Controls only the MiniMax H3 video-prompt rewrite. First-frame prompt and image "
                    "caching remain controlled by Caching Mode above. Fresh rewrites are retained for later reuse."
                ),
                scale=4,
            )
            clear_h3_prompt_cache_btn = gr.Button("🧹 Clear Cached H3 Prompts", scale=1)
        h3_prompt_cache_status = gr.Textbox(
            label="H3 Prompt Cache", interactive=False,
            visible=(config.VIDEO_BACKEND == "MiniMax H3"),
        )

        with gr.Accordion("✏️ Style Editor", open=False):
            gr.Markdown("Edit the style prompt for the selected style. Changes are saved per-project and never overwrite `styles.json`.")
            style_prompt_edit = gr.Textbox(
                label="Style Prompt Template (use `{prompt}` for your video prompt)",
                lines=3, interactive=False,
                placeholder='e.g. "A neon cyberpunk scene of {prompt}, vivid colors, glowing signs."'
            )
            style_negative_edit = gr.Textbox(
                label="Additional Negative Prompts",
                lines=2, interactive=False,
                placeholder="e.g. muted colors, realism"
            )

        with gr.Row():
            queue_pause_btn = gr.Button("⏸ Pause Queue")
            queue_cancel_btn = gr.Button("🧹 Cancel Pending")
            vid_gen_start_btn = gr.Button("🎬 Start Batch Generation", variant="primary", scale=2)
            vid_gen_stop_btn = gr.Button("⏹ Stop Active & Clear", variant="stop", visible=True)

        vid_gen_status = gr.Textbox(label="Queue Status", interactive=False, lines=8)

        with gr.Row():
            current_render_progress = gr.Slider(label="Current Render", minimum=0, maximum=100, value=0, interactive=False, step=1)
            queue_progress_bar = gr.Slider(label="Queue Progress", minimum=0, maximum=100, value=0, interactive=False, step=1)
        with gr.Row():
            current_render_eta = gr.Textbox(label="Render ETA", interactive=False, scale=1)
            queue_eta_txt = gr.Textbox(label="Queue Complete", interactive=False, scale=1)
            current_render_cost = gr.Textbox(label="Render Cost Est.", interactive=False, scale=1)
            queue_cost_txt = gr.Textbox(label="Queue Cost Est.", interactive=False, scale=1)

        gr.Markdown("### 🎯 Single Shot Generation")
        with gr.Row():
            single_shot_dropdown = gr.Dropdown(label="Select Shot to Generate", choices=[], interactive=True)
            single_shot_type_radio = gr.Radio(
                choices=["Vocal", "Action"],
                label="Shot Type",
                interactive=True,
                value=None,
                scale=1,
            )
            single_shot_btn = gr.Button("Generate Additional Version", variant="primary")
        with gr.Row():
            single_shot_camera_dropdown = gr.Dropdown(
                choices=config.CAMERA_MOTIONS, value="none",
                label="Camera Motion (this shot only)"
            )
        single_shot_prompt_edit = gr.Textbox(label="Edit Video Prompt for Selected Shot", lines=3, interactive=True)
        with gr.Row(visible=False) as first_frame_prompt_row:
            with gr.Column():
                first_frame_prompt_edit = gr.Textbox(
                    label="First Frame Image Prompt (Z-Image)",
                    lines=3, interactive=True,
                    placeholder="Leave blank to auto-generate via LLM on the next render, or type a custom image prompt."
                )
                first_frame_img_status = gr.Markdown(value="", visible=False)
                with gr.Row():
                    regen_first_frame_prompt_btn = gr.Button("🖼 Regenerate First Frame Prompt", size="sm")
                    clear_first_frame_img_btn = gr.Button("🗑 Clear Cached Frame", size="sm")
        with gr.Accordion("Full Prompt Text (character bibles + style + director injected)", open=False):
            full_prompt_preview = gr.Textbox(
                label="Full Prompt Text — edit below to override for this shot",
                interactive=True, lines=5, max_lines=30,
                placeholder="Select a shot to preview the fully assembled prompt. Edit freely, then click 'Save Override' to lock in your changes."
            )
            with gr.Row():
                save_override_btn = gr.Button("💾 Save Override", variant="primary", size="sm")
                clear_override_btn = gr.Button("🗑 Clear Override", variant="secondary", size="sm", visible=False)
            override_status = gr.Markdown("", visible=False)
        with gr.Row():
            prev_shot_btn_t3 = gr.Button("◀ Prev Shot")
            next_shot_btn_t3 = gr.Button("Next Shot ▶")

        with gr.Row():
            with gr.Column(scale=1):
                vid_gallery = gr.Gallery(label="Generated Video Thumbnails", columns=4, height=600, allow_preview=False, interactive=True)

            with gr.Column(scale=1):
                vid_large_view = gr.Video(label="Selected Video", interactive=False)
                with gr.Row():
                    del_vid_btn = gr.Button("🗑️ Delete This Video", variant="stop")
                with gr.Row():
                    regen_vid_same_prompt_btn = gr.Button("♻️ Regenerate Video (Same Prompt)")
                    regen_vid_new_prompt_btn = gr.Button("✨ Regenerate Video AND Prompt", variant="primary")

    # --- Tab 3 Internal Events ---

    def load_single_shot_prompt(shot_id, pm):
        if not shot_id or not pm.current_project or pm.df.empty: return ""
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            return ""
        row = pm.df.loc[row_idx[0]]
        val = row.get('Video_Prompt', '')
        return "" if pd.isna(val) else str(val)

    single_shot_dropdown.change(load_single_shot_prompt, inputs=[single_shot_dropdown, pm_state], outputs=[single_shot_prompt_edit])
    single_shot_dropdown.change(lambda s: s, inputs=[single_shot_dropdown], outputs=[shared_shot_state])

    def load_shot_type(shot_id, pm):
        if not shot_id or not pm.current_project or pm.df.empty:
            return gr.update(value=None)
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            return gr.update(value=None)
        return gr.update(value=pm.df.loc[row_idx[0], 'Type'])

    def save_shot_type(shot_id, new_type, pm):
        if not shot_id or not new_type or not pm.current_project or pm.df.empty:
            return
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx:
            pm.df.at[row_idx[0], 'Type'] = new_type
            pm.save_data()

    single_shot_dropdown.change(load_shot_type, inputs=[single_shot_dropdown, pm_state], outputs=[single_shot_type_radio])

    def load_first_frame_prompt(shot_id, pm):
        if not shot_id or pm.df.empty: return ""
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx and "First_Frame_Prompt" in pm.df.columns:
            val = pm.df.loc[row_idx[0], 'First_Frame_Prompt']
            return "" if pd.isna(val) else str(val)
        return ""

    single_shot_dropdown.change(load_first_frame_prompt, inputs=[single_shot_dropdown, pm_state], outputs=[first_frame_prompt_edit])

    def save_first_frame_prompt(shot_id, new_prompt, pm):
        if not shot_id or pm.df.empty: return
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx and "First_Frame_Prompt" in pm.df.columns:
            pm.df.at[row_idx[0], 'First_Frame_Prompt'] = new_prompt
            if "First_Frame_Prompt_Purpose" in pm.df.columns:
                pm.df.at[row_idx[0], "First_Frame_Prompt_Purpose"] = "manual" if str(new_prompt or "").strip() else ""
            if "First_Frame_Prompt_Source_Hash" in pm.df.columns:
                pm.df.at[row_idx[0], "First_Frame_Prompt_Source_Hash"] = ""
            # Prompt changed — clear any cached first frame image so a fresh one is generated
            for column in ("First_Frame_Image_Path", "First_Frame_Image_Source",
                           "First_Frame_Image_Aspect", "First_Frame_Image_Prompt_Hash"):
                if column in pm.df.columns:
                    pm.df.at[row_idx[0], column] = ""
            pm.save_data()

    first_frame_prompt_edit.change(save_first_frame_prompt, inputs=[single_shot_dropdown, first_frame_prompt_edit, pm_state])

    def get_first_frame_img_status(shot_id, pm):
        if not shot_id or pm.df.empty:
            return gr.update(visible=False)
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            return gr.update(visible=False)
        cached_rel = pm.df.loc[row_idx[0]].get("First_Frame_Image_Path", "")
        if not cached_rel or pd.isna(cached_rel) or not str(cached_rel).strip():
            return gr.update(visible=False)
        project_root = os.path.join(pm.base_dir, pm.current_project)
        abs_path = os.path.join(project_root, str(cached_rel).strip())
        if os.path.isfile(abs_path):
            return gr.update(value="✅ Cached frame image ready for reuse.", visible=True)
        else:
            return gr.update(value="⚠️ Cached frame image path recorded but file is missing — will regenerate.", visible=True)

    def handle_clear_first_frame_image(shot_id, pm):
        if not shot_id or pm.df.empty:
            gr.Warning("No shot selected.")
            return gr.update(visible=False)
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            gr.Warning(f"Shot {shot_id} not found.")
            return gr.update(visible=False)
        changed = False
        if "First_Frame_Image_Path" in pm.df.columns and pm.df.at[row_idx[0], 'First_Frame_Image_Path']:
            pm.df.at[row_idx[0], 'First_Frame_Image_Path'] = ""
            changed = True
        for column in ("First_Frame_Image_Source", "First_Frame_Image_Aspect",
                       "First_Frame_Image_Prompt_Hash"):
            if column in pm.df.columns:
                pm.df.at[row_idx[0], column] = ""
        if changed:
            pm.save_data()
            gr.Info(f"Cached first frame image cleared for {shot_id}. A fresh image will be generated on next render.")
        else:
            gr.Info(f"No cached image to clear for {shot_id}.")
        return gr.update(visible=False)

    single_shot_dropdown.change(get_first_frame_img_status, inputs=[single_shot_dropdown, pm_state], outputs=[first_frame_img_status])

    def toggle_firstframe_controls(mode, backend):
        uses_generated_frame = mode in ("Krea 2 First Frame", "Z-Image First Frame")
        show_zimage_backend = mode == "Z-Image First Frame" and backend != "MiniMax H3"
        return (gr.update(visible=uses_generated_frame), gr.update(visible=uses_generated_frame),
                gr.update(visible=uses_generated_frame),
                gr.update(visible=False) if not uses_generated_frame else gr.update(),
                gr.update(visible=show_zimage_backend))

    vid_firstframe_mode.change(
        toggle_firstframe_controls,
        inputs=[vid_firstframe_mode, backend_state],
        outputs=[llm_image_prompt_dropdown, first_frame_prompt_row, first_frame_reuse_dropdown, first_frame_img_status, vid_zimage_backend]
    )

    clear_first_frame_img_btn.click(
        handle_clear_first_frame_image,
        inputs=[single_shot_dropdown, pm_state],
        outputs=[first_frame_img_status]
    )

    def _nav_shot_t3(new_shot, gallery_paths):
        shot_vid = next(
            (p for p in gallery_paths if os.path.basename(p).split('_')[0] == new_shot),
            None
        )
        return gr.update(value=new_shot), shot_vid, shot_vid or ""

    def get_prev_shot_t3(current_shot, pm, gallery_paths):
        if pm.df.empty: return gr.update(), None, ""
        choices = pm.df['Shot_ID'].dropna().unique().tolist()
        if not choices: return gr.update(), None, ""
        if current_shot not in choices:
            new_shot = choices[-1]
        else:
            new_shot = choices[(choices.index(current_shot) - 1) % len(choices)]
        return _nav_shot_t3(new_shot, gallery_paths)

    def get_next_shot_t3(current_shot, pm, gallery_paths):
        if pm.df.empty: return gr.update(), None, ""
        choices = pm.df['Shot_ID'].dropna().unique().tolist()
        if not choices: return gr.update(), None, ""
        if current_shot not in choices:
            new_shot = choices[0]
        else:
            new_shot = choices[(choices.index(current_shot) + 1) % len(choices)]
        return _nav_shot_t3(new_shot, gallery_paths)

    prev_shot_btn_t3.click(
        get_prev_shot_t3,
        inputs=[single_shot_dropdown, pm_state, gallery_paths_state],
        outputs=[single_shot_dropdown, vid_large_view, selected_vid_path]
    )
    next_shot_btn_t3.click(
        get_next_shot_t3,
        inputs=[single_shot_dropdown, pm_state, gallery_paths_state],
        outputs=[single_shot_dropdown, vid_large_view, selected_vid_path]
    )

    def save_single_shot_prompt(shot_id, new_prompt, pm):
        if not shot_id or pm.df.empty: return
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx:
            pm.df.at[row_idx[0], 'Video_Prompt'] = new_prompt
            # Clear stale cached first-frame prompt whenever the video prompt changes
            if "First_Frame_Prompt" in pm.df.columns:
                pm.df.at[row_idx[0], 'First_Frame_Prompt'] = ""
            for column in ("First_Frame_Prompt_Purpose", "First_Frame_Prompt_Source_Hash"):
                if column in pm.df.columns:
                    pm.df.at[row_idx[0], column] = ""
            # Clear override — base prompt changed so override text is now stale
            pm.df.at[row_idx[0], 'Prompt_Override'] = ''
            pm.df.at[row_idx[0], 'Prompt_Override_Text'] = ''
            pm.save_data()

    single_shot_prompt_edit.change(save_single_shot_prompt, inputs=[single_shot_dropdown, single_shot_prompt_edit, pm_state])

    def build_full_prompt_preview(shot_id, style, director, vocal_mode, pm):
        if not shot_id or pm.df.empty:
            return "", gr.update(visible=False), gr.update(visible=False)
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            return "", gr.update(visible=False), gr.update(visible=False)
        row = pm.df.loc[row_idx[0]]

        is_override = str(row.get('Prompt_Override', '')).strip().lower() == 'true'
        if is_override:
            override_text = str(row.get('Prompt_Override_Text', '')).strip()
            return (
                override_text,
                gr.update(value="⚡ **Override active** — this shot renders with the text above, bypassing character bible injection, style wrapping, and director credit.", visible=True),
                gr.update(visible=True),
            )

        vid_prompt_raw = row.get('Video_Prompt', '')
        vid_prompt = "" if pd.isna(vid_prompt_raw) else str(vid_prompt_raw).strip()

        if row.get('Type') == "Vocal" and vocal_mode == "Use Singer/Band Description":
            settings = pm.load_project_settings()
            perf_desc = settings.get("performance_desc", "")
            if perf_desc:
                vid_prompt = perf_desc

        if not vid_prompt:
            return "", gr.update(visible=False), gr.update(visible=False)

        if pm.character_bibles:
            vid_prompt = apply_character_bibles(vid_prompt, pm.character_bibles)

        style_data = resolve_style_data(style, pm)
        if style_data:
            vid_prompt = style_data["prompt"].replace("{prompt}", vid_prompt)

        if director and director != "None":
            effective_director = director
            if director == "Custom":
                settings = pm.load_project_settings()
                effective_director = settings.get("custom_director", "")
            if effective_director:
                vid_prompt += f". This video was directed by {effective_director}."

        return vid_prompt, gr.update(visible=False), gr.update(visible=False)

    _full_prompt_inputs = [single_shot_dropdown, vid_style_dropdown, vid_director_dropdown, vid_vocal_prompt_mode, pm_state]
    _full_prompt_outputs = [full_prompt_preview, override_status, clear_override_btn]
    single_shot_dropdown.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)
    single_shot_prompt_edit.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)
    # Shot type radio — save to CSV and refresh full prompt preview (Vocal type uses performance_desc)
    single_shot_type_radio.change(save_shot_type, inputs=[single_shot_dropdown, single_shot_type_radio, pm_state])
    single_shot_type_radio.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)
    vid_style_dropdown.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)
    vid_director_dropdown.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)
    vid_custom_director_txt.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)
    vid_vocal_prompt_mode.change(build_full_prompt_preview, inputs=_full_prompt_inputs, outputs=_full_prompt_outputs)

    def save_prompt_override(shot_id, override_text, pm):
        if not shot_id or pm.df.empty:
            return gr.update(), gr.update()
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx:
            pm.df.at[row_idx[0], 'Prompt_Override'] = 'True'
            pm.df.at[row_idx[0], 'Prompt_Override_Text'] = override_text.strip()
            pm.save_data()
            gr.Info(f"Override saved for {shot_id}.")
        return (
            gr.update(value="⚡ **Override active** — this shot renders with the text above, bypassing character bible injection, style wrapping, and director credit.", visible=True),
            gr.update(visible=True),
        )

    save_override_btn.click(
        save_prompt_override,
        inputs=[single_shot_dropdown, full_prompt_preview, pm_state],
        outputs=[override_status, clear_override_btn],
    )

    def clear_prompt_override(shot_id, style, director, vocal_mode, pm):
        if not shot_id or pm.df.empty:
            return gr.update(), gr.update(), gr.update()
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx:
            pm.df.at[row_idx[0], 'Prompt_Override'] = ''
            pm.df.at[row_idx[0], 'Prompt_Override_Text'] = ''
            pm.save_data()
            gr.Info(f"Override cleared for {shot_id}.")
        assembled = build_full_prompt_preview(shot_id, style, director, vocal_mode, pm)
        return assembled[0], gr.update(visible=False), gr.update(visible=False)

    clear_override_btn.click(
        clear_prompt_override,
        inputs=[single_shot_dropdown, vid_style_dropdown, vid_director_dropdown, vid_vocal_prompt_mode, pm_state],
        outputs=[full_prompt_preview, override_status, clear_override_btn],
    )

    # --- Style Editor handlers ---

    def on_style_change(style_name, pm):
        if not style_name or style_name == "None":
            return gr.update(value="", interactive=False), gr.update(value="", interactive=False)
        settings = pm.load_project_settings() if pm.current_project else {}
        if style_name == "Custom":
            p = settings.get("custom_style_prompt", "")
            n = settings.get("custom_style_negative", "")
            return gr.update(value=p, interactive=True), gr.update(value=n, interactive=True)
        overrides = settings.get("style_overrides", {})
        if style_name in overrides:
            p = overrides[style_name].get("prompt", "")
            n = overrides[style_name].get("negative_prompt", "")
        else:
            style = next((s for s in config.STYLES if s["name"] == style_name), None)
            p = style["prompt"] if style else ""
            n = style["negative_prompt"] if style else ""
        return gr.update(value=p, interactive=True), gr.update(value=n, interactive=True)

    def save_style_prompt(style_name, prompt_val, pm):
        if not style_name or style_name == "None" or not pm.current_project:
            return
        if style_name == "Custom":
            pm.save_project_settings({"custom_style_prompt": prompt_val})
        else:
            settings = pm.load_project_settings()
            overrides = settings.get("style_overrides", {})
            if style_name not in overrides:
                overrides[style_name] = {}
            overrides[style_name]["prompt"] = prompt_val
            pm.save_project_settings({"style_overrides": overrides})

    def save_style_negative(style_name, negative_val, pm):
        if not style_name or style_name == "None" or not pm.current_project:
            return
        if style_name == "Custom":
            pm.save_project_settings({"custom_style_negative": negative_val})
        else:
            settings = pm.load_project_settings()
            overrides = settings.get("style_overrides", {})
            if style_name not in overrides:
                overrides[style_name] = {}
            overrides[style_name]["negative_prompt"] = negative_val
            pm.save_project_settings({"style_overrides": overrides})

    vid_style_dropdown.change(on_style_change, inputs=[vid_style_dropdown, pm_state], outputs=[style_prompt_edit, style_negative_edit])
    style_prompt_edit.change(save_style_prompt, inputs=[vid_style_dropdown, style_prompt_edit, pm_state])
    style_negative_edit.change(save_style_negative, inputs=[vid_style_dropdown, style_negative_edit, pm_state])

    # --- Director handlers ---

    def on_director_change(director_name, pm):
        if director_name == "Custom":
            saved = pm.load_project_settings().get("custom_director", "") if pm.current_project else ""
            return gr.update(visible=True, value=saved)
        return gr.update(visible=False, value="")

    def save_custom_director(name, pm):
        if pm.current_project:
            pm.save_project_settings({"custom_director": name})

    vid_director_dropdown.change(on_director_change, inputs=[vid_director_dropdown, pm_state], outputs=[vid_custom_director_txt])
    vid_custom_director_txt.change(save_custom_director, inputs=[vid_custom_director_txt, pm_state])

    # --- LoRA list fetching ---

    def _fetch_lora_choices():
        # Response: { "loras": [{ "name": str, "path": str }], "loras_dir": str, "error": null }
        try:
            import requests as _req
            resp = _req.get(f"{config.LTX_BASE_URL}/loras", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            loras = data.get("loras", []) if isinstance(data, dict) else data
            paths = [entry["path"] for entry in loras if isinstance(entry, dict) and "path" in entry]
            if paths:
                return ["None"] + paths
        except Exception:
            pass
        return ["None"]

    vid_lora_refresh_btn.click(
        lambda: gr.update(choices=_fetch_lora_choices()),
        outputs=[vid_lora_dropdown],
    )

    # --- Persist Tab 3 preferences to project settings ---

    def auto_save_tab3_prefs(firstframe, llm_img, reuse, h3_prompt_cache_mode,
                              zimage_backend, vocal_mode, gen_mode,
                              versions, resolution, camera, director, style, vocal_chain, lora, pm):
        if not pm or not pm.current_project:
            return
        pm.save_project_settings({
            "firstframe_mode": firstframe,
            "llm_image_prompt_mode": llm_img,
            "first_frame_reuse_mode": reuse,
            "h3_prompt_cache_mode": h3_prompt_cache_mode,
            "zimage_backend": zimage_backend,
            "vocal_prompt_mode": vocal_mode,
            "last_gen_mode": gen_mode,
            "last_versions": versions,
            "last_resolution": resolution,
            "last_camera_motion": camera,
            "last_director": director,
            "last_style": style,
            "vocal_chain_mode": vocal_chain,
            "last_lora": lora,
        })

    _tab3_pref_inputs = [
        vid_firstframe_mode, llm_image_prompt_dropdown, first_frame_reuse_dropdown,
        h3_prompt_cache_dropdown,
        vid_zimage_backend, vid_vocal_prompt_mode, vid_gen_mode_dropdown, vid_versions_dropdown,
        vid_resolution_dropdown, single_shot_camera_dropdown,
        vid_director_dropdown, vid_style_dropdown, vid_vocal_chain_checkbox, vid_lora_dropdown, pm_state,
    ]
    for _t3_comp in [vid_firstframe_mode, llm_image_prompt_dropdown, first_frame_reuse_dropdown,
                     h3_prompt_cache_dropdown,
                     vid_zimage_backend, vid_vocal_prompt_mode, vid_gen_mode_dropdown, vid_versions_dropdown,
                     vid_resolution_dropdown, single_shot_camera_dropdown,
                     vid_director_dropdown, vid_style_dropdown, vid_vocal_chain_checkbox, vid_lora_dropdown]:
        _t3_comp.change(auto_save_tab3_prefs, inputs=_tab3_pref_inputs)

    def clear_h3_prompt_cache(pm):
        if not pm or not pm.current_project:
            return "❌ Load a project before clearing its H3 prompt cache."
        if getattr(pm, "queue_processor_running", False):
            return "⚠️ Stop the active render queue before clearing cached H3 prompts."
        with getattr(pm, "queue_lock", threading.Lock()):
            settings = pm.load_project_settings()
            cache = settings.get("h3_prompt_cache", {})
            count = len(cache) if isinstance(cache, dict) else 0
            pm.save_project_settings({"h3_prompt_cache": {}})
        return f"✅ Cleared {count} cached H3 prompt{'s' if count != 1 else ''}."

    clear_h3_prompt_cache_btn.click(
        clear_h3_prompt_cache, inputs=[pm_state], outputs=[h3_prompt_cache_status]
    )

    def save_h3_preferences(aspect, quality, lipsync_output, custom_width, custom_height, pm):
        if pm and pm.current_project:
            pm.save_project_settings({
                "h3_aspect": aspect,
                "h3_quality": quality,
                "h3_lipsync_output": lipsync_output,
                "h3_custom_width": custom_width,
                "h3_custom_height": custom_height,
            })
            expected_aspect = h3_aspect_cache_key(aspect, custom_width, custom_height)
            if "First_Frame_Image_Aspect" in pm.df.columns:
                invalid = pm.df["First_Frame_Image_Path"].astype(str).str.strip().ne("") & (
                    pm.df["First_Frame_Image_Aspect"].astype(str) != expected_aspect
                )
                if invalid.any():
                    for column in ("First_Frame_Image_Path", "First_Frame_Image_Source",
                                   "First_Frame_Image_Aspect", "First_Frame_Image_Prompt_Hash"):
                        if column in pm.df.columns:
                            pm.df.loc[invalid, column] = ""
                    pm.save_data()

    for _h3_comp in [h3_aspect_dropdown, h3_quality_dropdown, h3_lipsync_output_dropdown, h3_custom_width, h3_custom_height]:
        _h3_comp.change(
            save_h3_preferences,
            inputs=[h3_aspect_dropdown, h3_quality_dropdown, h3_lipsync_output_dropdown, h3_custom_width, h3_custom_height, pm_state],
        )

    h3_aspect_dropdown.change(
        lambda aspect: (gr.update(visible=(aspect == "CUSTOM")), gr.update(visible=(aspect == "CUSTOM"))),
        inputs=[h3_aspect_dropdown],
        outputs=[h3_custom_width, h3_custom_height],
    )

    # --- Gallery select ---

    def on_vid_gallery_select(evt: gr.SelectData, gallery_paths):
        if gallery_paths and evt.index < len(gallery_paths):
            fpath = gallery_paths[evt.index]
            fname = os.path.basename(fpath)
            shot_id = fname.split('_')[0] if '_' in fname else "Unknown"
            return fpath, fpath, gr.update(value=shot_id)
        return None, "", gr.update()

    vid_gallery.select(on_vid_gallery_select, inputs=[gallery_paths_state], outputs=[vid_large_view, selected_vid_path, single_shot_dropdown])

    def update_single_shot_choices(pm, shared_shot):
        if pm.df.empty: return gr.update(choices=[], value=None), None
        choices = pm.df['Shot_ID'].dropna().unique().tolist()
        value = shared_shot if shared_shot in choices else None
        return gr.update(choices=choices, value=value), shared_shot

    def format_queue_status(pm, current_item=None, current_msg=""):
        runtime = getattr(pm, "pipeline_runtime", None)
        if runtime and runtime.is_active():
            return runtime.status_text()
        lines = []
        if config.RENDER_QUEUE_MODE == "Pipelined" and config.VIDEO_BACKEND != "MiniMax H3":
            lines.append("ℹ️ Pipelined staging is MiniMax H3-only; this backend is using the legacy queue.")
        if getattr(pm, 'ltx_ram_warning', ''):
            lines.append(pm.ltx_ram_warning)
        if current_item:
            shot_label = f"{current_item['shot_id']} — {current_item['resolution']} — {current_item['style']}"
            director = current_item.get('director', 'None')
            if director and director != 'None':
                shot_label += f" — dir: {director}"
            lines.append(f"🎬 NOW RENDERING: {shot_label}")
            if current_msg:
                lines.append(f"  ⏳ {current_msg}")
            # Show run-ahead status
            if getattr(pm, '_runhead_running', False):
                _rh_sid = getattr(pm, '_runhead_shot_id', '?')
                lines.append(f"  🚀 Run-ahead: generating Z-Image for {_rh_sid}...")
            elif getattr(pm, '_runhead_result', None):
                _rh_sid = pm._runhead_result[0]
                lines.append(f"  ✅ Run-ahead: Z-Image for {_rh_sid} ready")
        if pm.render_queue:
            lines.append(f"📋 QUEUE ({len(pm.render_queue)}):")
            for i, item in enumerate(pm.render_queue, 1):
                lines.append(f"  {i}. {item['shot_id']} — {item['resolution']} — {item['style']}")
        if not lines:
            if getattr(pm, 'last_gen_error', ''):
                return f"💤 Queue is empty.\n❌ {pm.last_gen_error}"
            return "💤 Queue is empty."
        if getattr(pm, 'last_gen_error', ''):
            lines.append(f"❌ {pm.last_gen_error}")
        return "\n".join(lines)

    def process_pipelined_queue_if_idle(pm, proj):
        with pm.queue_lock:
            if pm.queue_processor_running:
                gal = get_project_videos(pm, proj)
                runtime = getattr(pm, "pipeline_runtime", None)
                status = runtime.status_text() if runtime else "🚦 Pipelined queue is already running."
                yield gal, status, [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
                return
            items = list(pm.render_queue)
            pm.render_queue.clear()
            if not items:
                gal = get_project_videos(pm, proj)
                yield gal, "💤 Queue is empty.", [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
                return
            pm.queue_processor_running = True
            pm.stop_video_generation = False

        runtime = H3RenderPipeline(
            pm, proj, llm_concurrency=config.LLM_CONCURRENCY,
            coop_enabled=config.H3_COOP_ENABLED,
            min_available_commit_gb=config.H3_COOP_MIN_AVAILABLE_COMMIT_GB,
        )
        pm.pipeline_runtime = runtime
        runtime.start(items)
        try:
            while runtime.is_active():
                snap = runtime.snapshot()
                gal = get_project_videos(pm, proj)
                eta = f"~{format_eta(snap['eta_seconds'])}" if snap["eta_seconds"] is not None else "Learning throughput..."
                elapsed = max(0.0, time.time() - runtime.started_at)
                cost = (elapsed / 3600.0) * (config.SYSTEM_WATTAGE / 1000.0) * config.ELECTRICITY_COST
                active_video = bool(snap["active"]["video"])
                yield (
                    gal, runtime.status_text(), [item[0] for item in gal],
                    50 if active_video else 0, snap["progress"], "" if not active_video else "Rendering...",
                    eta, f"${cost:.4f}" if active_video else "", f"${cost:.3f}",
                    gr.update(), gr.update(), gr.update(),
                )
                time.sleep(0.5)
        finally:
            runtime.thread.join(timeout=2) if runtime.thread else None
            sync_video_directory(pm)
            with pm.queue_lock:
                pm.queue_processor_running = False
                pm.stop_video_generation = False
            gal = get_project_videos(pm, proj)
            snap = runtime.snapshot()
            final = runtime.status_text()
            if snap["failed"]:
                final += f"\n⚠️ Finished with {snap['failed']} failed/cancelled job(s)."
            else:
                final += "\n✅ Pipelined queue complete."
            yield gal, final, [item[0] for item in gal], 0, 100 if snap["total"] else 0, "", "", "", "", gr.update(), gr.update(), gr.update()

    def _effective_resolution(shot_id, resolution, df):
        """Downgrade 1080p to 720p for shots longer than 5 seconds.
        LTX Desktop only supports >5s clips at 720p or lower."""
        if resolution != "1080p":
            return resolution
        try:
            dur = float(df[df['Shot_ID'] == shot_id]['Duration'].values[0])
            if dur > 5.0 + (1 / 24):  # LTX snaps 5s → 5+1/24; only downgrade truly >5s shots
                return "720p"
        except Exception:
            pass
        return resolution

    def add_to_render_queue(shot_id, resolution, vocal_mode, style, director, generation_mode, pm, delete_path=None, camera_motion="none", use_llm_image_prompt=False, caching_mode="Use cached prompt", h3_prompt_cache_mode="Reuse cached H3 prompts", vocal_chain_mode=False, zimage_backend="LTX Desktop", lora_path="None"):
        if not shot_id:
            return "❌ No shot selected."
        effective_res = _effective_resolution(shot_id, resolution, pm.df)
        item = {'shot_id': shot_id, 'resolution': effective_res, 'vocal_mode': vocal_mode,
                'style': style, 'director': director, 'generation_mode': generation_mode,
                'delete_path': delete_path, 'camera_motion': camera_motion,
                'use_llm_image_prompt': use_llm_image_prompt,
                'caching_mode': caching_mode,
                'h3_prompt_cache_mode': h3_prompt_cache_mode,
                'vocal_chain_mode': vocal_chain_mode,
                'zimage_backend': zimage_backend,
                'lora_path': lora_path}
        with pm.queue_lock:
            pm.render_queue.append(item)
        status = format_queue_status(pm)
        if effective_res != resolution:
            status = f"⚠️ Shot {shot_id} is >5s — resolution downgraded to 720p.\n" + status
        return status

    _queue_est_cache = [0.0, -1]  # [cached_estimate, last_queue_len]
    _jit_state = [False]  # [is_jit_converting] — shared mutable flag for JIT thread

    def _calc_remaining_queue_est(pm):
        """Dynamically sum estimates for all items still in pm.render_queue.
        Caches by queue length to avoid redundant DataFrame scans on every 1s yield."""
        with pm.queue_lock:
            remaining = list(pm.render_queue)
        queue_len = len(remaining)
        if queue_len == _queue_est_cache[1]:
            return _queue_est_cache[0]
        est = 0.0
        for _qi in remaining:
            try:
                _dur = float(pm.df[pm.df['Shot_ID'] == _qi['shot_id']]['Duration'].values[0])
            except Exception:
                _dur = 3.0
            est += config.estimate_render_seconds(_dur, _qi['resolution'], _qi.get('generation_mode', 'LTX-Native'))
        _queue_est_cache[0] = est
        _queue_est_cache[1] = queue_len
        return est

    def process_render_queue_if_idle(pm, proj):
        if config.RENDER_QUEUE_MODE == "Pipelined" and config.VIDEO_BACKEND == "MiniMax H3":
            yield from process_pipelined_queue_if_idle(pm, proj)
            return
        with pm.queue_lock:
            if pm.queue_processor_running or not pm.render_queue:
                gal = get_project_videos(pm, proj)
                yield gal, format_queue_status(pm), [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
                return
            pm.queue_processor_running = True
            pm.stop_video_generation = False
            pm.last_gen_error = ""

        queue_elapsed = 0.0
        _first_shot_in_session = True

        # Run-ahead state for ComfyUI-run-ahead mode
        pm._runhead_running = False
        pm._runhead_shot_id = None
        pm._runhead_result = None
        pm._runhead_thread = None

        try:
            while True:
                if pm.queue_paused:
                    with pm.queue_lock:
                        queue_snapshot = list(pm.render_queue)
                    gal = get_project_videos(pm, proj)
                    paused_lines = ["⏸ [PAUSED]"]
                    if queue_snapshot:
                        paused_lines.append(f"📋 QUEUE ({len(queue_snapshot)}):")
                        for i, it in enumerate(queue_snapshot, 1):
                            paused_lines.append(f"  {i}. {it['shot_id']} — {it['resolution']} — {it['style']}")
                    yield gal, "\n".join(paused_lines), [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
                    time.sleep(0.5)
                    continue

                with pm.queue_lock:
                    if not pm.render_queue or pm.stop_video_generation:
                        break
                    current_item = pm.render_queue.pop(0)

                delete_path = current_item.get('delete_path')
                if delete_path and os.path.exists(delete_path):
                    try:
                        os.remove(delete_path)
                    except Exception as e:
                        print(f"Could not delete {delete_path}: {e}")

                # Duration and time estimate for this shot
                try:
                    shot_dur = float(pm.df[pm.df['Shot_ID'] == current_item['shot_id']]['Duration'].values[0])
                except Exception:
                    shot_dur = 3.0
                shot_est = config.estimate_render_seconds(shot_dur, current_item['resolution'], current_item.get('generation_mode', 'LTX-Native'))
                render_start = time.time()

                _jit_started_for_this_shot = False
                _runhead_started_for_this_shot = False
                pm._display_ffp = None  # reset side-channel before each shot

                for path, msg in generate_video_for_shot(
                    current_item['shot_id'], current_item['resolution'],
                    current_item['vocal_mode'], pm, current_item['style'],
                    director=current_item.get('director'),
                    generation_mode=current_item.get('generation_mode', 'LTX-Native'),
                    camera_motion=current_item.get('camera_motion', 'none'),
                    use_llm_image_prompt=current_item.get('use_llm_image_prompt', False),
                    caching_mode=current_item.get('caching_mode', 'Use cached prompt'),
                    h3_prompt_cache_mode=current_item.get(
                        'h3_prompt_cache_mode', 'Reuse cached H3 prompts'
                    ),
                    vocal_chain_mode=current_item.get('vocal_chain_mode', False),
                    zimage_backend=current_item.get('zimage_backend', 'LTX Desktop'),
                    lora_path=current_item.get('lora_path', 'None'),
                ):
                    if path is None:
                        if msg and msg.startswith("Error:"):
                            pm.last_gen_error = msg
                        ltx_pct = 0
                        _m = re.search(r'(\d+)%', msg)
                        if _m:
                            ltx_pct = int(_m.group(1))
                        elapsed = time.time() - render_start
                        # Dynamic estimates — accounts for shots added/removed mid-queue
                        remaining_queue_est = _calc_remaining_queue_est(pm)
                        render_remaining = max(0.0, shot_est - elapsed)
                        queue_remaining = render_remaining + remaining_queue_est
                        total_done = queue_elapsed + elapsed
                        total_dynamic_est = total_done + queue_remaining
                        queue_pct = min(100, int(total_done / max(total_dynamic_est, 1) * 100))
                        render_pct = min(99, int(elapsed / max(shot_est, 1) * 100))
                        render_cost = (elapsed / 3600.0) * (config.SYSTEM_WATTAGE / 1000.0) * config.ELECTRICITY_COST
                        queue_cost_proj = (total_dynamic_est / 3600.0) * (config.SYSTEM_WATTAGE / 1000.0) * config.ELECTRICITY_COST
                        gal = get_project_videos(pm, proj)
                        # Consume any pending first-frame prompt set by generate_video_for_shot
                        _ffp_update = gr.update()
                        _pending_ffp = getattr(pm, '_display_ffp', None)
                        if _pending_ffp is not None:
                            _ffp_update = _pending_ffp
                            pm._display_ffp = None
                        yield (gal, format_queue_status(pm, current_item, msg), [item[0] for item in gal],
                               render_pct, queue_pct,
                               f"~{format_eta(render_remaining)}", f"~{format_eta(queue_remaining)}",
                               f"${render_cost:.4f}", f"${queue_cost_proj:.3f}",
                               gr.update(), gr.update(), _ffp_update)

                        # JIT: while the current video is rendering, pre-convert the next shot's
                        # image prompt via LLM (LTX is busy; LM Studio is free).
                        # Skip JIT if next shot uses ComfyUI-run-ahead — the run-ahead worker handles everything.
                        if not _jit_started_for_this_shot and not _jit_state[0]:
                            with pm.queue_lock:
                                _next_items = list(pm.render_queue[:1])
                            if _next_items:
                                _ni = _next_items[0]
                                _next_is_runhead = (_ni.get('zimage_backend') == 'ComfyUI-run-ahead')
                                if (not _next_is_runhead
                                        and _ni.get('generation_mode') == 'Z-Image First Frame'
                                        and _ni.get('use_llm_image_prompt')):
                                    _next_row = pm.df[pm.df['Shot_ID'] == _ni['shot_id']]
                                    _cached = ""
                                    if not _next_row.empty and "First_Frame_Prompt" in pm.df.columns:
                                        _raw = _next_row.iloc[0].get("First_Frame_Prompt", "")
                                        if _raw and not pd.isna(_raw) and str(_raw).strip():
                                            _cached = str(_raw).strip()
                                    if not _cached:
                                        _jit_state[0] = True
                                        _jit_started_for_this_shot = True
                                        _ni_copy = dict(_ni)

                                        def _jit_worker(_ni_copy=_ni_copy):
                                            try:
                                                _s = pm.load_project_settings()
                                                _sid = _ni_copy['shot_id']
                                                _nr = pm.df[pm.df['Shot_ID'] == _sid]
                                                if _nr.empty:
                                                    return
                                                # Early exit checks based on caching mode
                                                _cmode = _ni_copy.get('caching_mode', 'Use cached prompt')
                                                # "Regenerate both": JIT pre-conversion is wasted — cache will be ignored at render time
                                                if _cmode == 'Regenerate both on each render':
                                                    print(f"⏭️ [JIT] Skipping — caching mode is 'Regenerate both' for {_sid}")
                                                    return
                                                # "Use cached image": if a valid image file already exists,
                                                # generate_video_for_shot will reuse it — no LLM pre-conversion needed
                                                if _cmode == 'Use cached image' and "First_Frame_Image_Path" in pm.df.columns:
                                                    _ridx_early = pm.df.index[pm.df['Shot_ID'] == _sid].tolist()
                                                    if _ridx_early:
                                                        _cached_rel = pm.df.loc[_ridx_early[0]].get("First_Frame_Image_Path", "")
                                                        if _cached_rel and not pd.isna(_cached_rel) and str(_cached_rel).strip():
                                                            _project_root = os.path.join(pm.base_dir, pm.current_project)
                                                            _abs = os.path.join(_project_root, str(_cached_rel).strip())
                                                            if os.path.isfile(_abs):
                                                                print(f"⏭️ [JIT] Skipping — valid cached first-frame image found for {_sid}")
                                                                return
                                                _type = _nr.iloc[0].get("Type", "")
                                                _vm = _ni_copy.get('vocal_mode', '')
                                                _style = _ni_copy.get('style')
                                                _director = _ni_copy.get('director')
                                                _style_data = resolve_style_data(_style, pm) if _style and _style != "None" else None

                                                def _jit_assemble(_base_p):
                                                    _p = _base_p
                                                    if pm.character_bibles:
                                                        _p = apply_character_bibles(_p, pm.character_bibles)
                                                    if _style_data:
                                                        _p = _style_data["prompt"].replace("{prompt}", _p)
                                                    if _director and _director != "None":
                                                        _eff = _director
                                                        if _director == "Custom":
                                                            _eff = _s.get("custom_director", "")
                                                        if _eff:
                                                            _p += f". This video was directed by {_eff}."
                                                    return _p

                                                if _type == "Vocal" and _vm == "Use Singer/Band Description":
                                                    _src = _s.get("performance_desc", "")
                                                    _assembled_vocal = _jit_assemble(_src) if _src else ""
                                                    if not _assembled_vocal:
                                                        return
                                                    if (_s.get("zimage_vocal_source_assembled") == _assembled_vocal
                                                            and _s.get("zimage_vocal_first_frame_prompt")):
                                                        return  # Already project-cached with matching assembled prompt
                                                    _base = _assembled_vocal
                                                else:
                                                    _vp = _nr.iloc[0].get("Video_Prompt", "")
                                                    _raw = "" if pd.isna(_vp) else str(_vp).strip()
                                                    _base = _jit_assemble(_raw) if _raw else ""
                                                if not _base:
                                                    return
                                                print(f"🧠 [JIT] Pre-converting first-frame prompt for {_sid}...")
                                                _converted = convert_prompt_for_zimage(_base, pm, _s)
                                                _ridx = pm.df.index[pm.df['Shot_ID'] == _sid].tolist()
                                                if _ridx and "First_Frame_Prompt" in pm.df.columns:
                                                    with pm.queue_lock:
                                                        pm.df.at[_ridx[0], "First_Frame_Prompt"] = _converted
                                                        if "First_Frame_Prompt_Purpose" in pm.df.columns:
                                                            if config.VIDEO_BACKEND == "MiniMax H3" and _type == "Vocal" and _vm == "Use Storyboard Prompt":
                                                                purpose = "h3_vocal_storyboard"
                                                            elif config.VIDEO_BACKEND == "MiniMax H3":
                                                                purpose = "h3_action_target"
                                                            else:
                                                                purpose = "zimage_generated"
                                                            pm.df.at[_ridx[0], "First_Frame_Prompt_Purpose"] = purpose
                                                        if "First_Frame_Prompt_Source_Hash" in pm.df.columns:
                                                            pm.df.at[_ridx[0], "First_Frame_Prompt_Source_Hash"] = hashlib.sha256(_base.encode("utf-8")).hexdigest()
                                                        pm.save_data()
                                                if _type == "Vocal" and _vm == "Use Singer/Band Description":
                                                    pm.save_project_settings({
                                                        "zimage_vocal_first_frame_prompt": _converted,
                                                        "zimage_vocal_source_assembled": _assembled_vocal,
                                                    })
                                                print(f"✅ [JIT] First-frame prompt cached for {_sid}")
                                            finally:
                                                _jit_state[0] = False

                                        threading.Thread(target=_jit_worker, daemon=True).start()

                        # RUN-AHEAD: while the current video renders on LTX, generate the next shot's
                        # Z-image first frame on ComfyUI (separate GPU). Skipped for chain-conditioned
                        # shots (they use the predecessor's look-ahead frame, not a Z-image) and for
                        # shots that already have a valid cached first-frame image.
                        if not _runhead_started_for_this_shot and not pm._runhead_running:
                            with pm.queue_lock:
                                _rh_next = list(pm.render_queue[:1])
                            if _rh_next:
                                _rh_ni = _rh_next[0]
                                if (
                                    _rh_ni.get('generation_mode') == 'Z-Image First Frame'
                                    and _rh_ni.get('zimage_backend') == 'ComfyUI-run-ahead'
                                    and not _has_valid_cached_first_frame(_rh_ni, pm)
                                ):
                                    pm._runhead_running = True
                                    pm._runhead_shot_id = _rh_ni['shot_id']
                                    # Do NOT clear pm._runhead_result here — it may hold a valid
                                    # pre-generated frame for the current shot that video.py
                                    # hasn't consumed yet. The worker overwrites it when done.
                                    _rh_ni_copy = dict(_rh_ni)

                                    def _runhead_worker(_rh_ni_copy=_rh_ni_copy):
                                        try:
                                            _s = pm.load_project_settings()
                                            _sid = _rh_ni_copy['shot_id']
                                            _nr = pm.df[pm.df['Shot_ID'] == _sid]
                                            if _nr.empty:
                                                return
                                            _cmode = _rh_ni_copy.get('caching_mode', 'Use cached prompt')
                                            if _cmode == 'Regenerate both on each render':
                                                pass  # Always regenerate — proceed with generation
                                            # Build assembled prompt (same logic as JIT worker)
                                            _type = _nr.iloc[0].get("Type", "")
                                            _vm = _rh_ni_copy.get('vocal_mode', '')
                                            _style = _rh_ni_copy.get('style')
                                            _director = _rh_ni_copy.get('director')
                                            _style_data = resolve_style_data(_style, pm) if _style and _style != "None" else None

                                            def _rh_assemble(_base_p):
                                                _p = _base_p
                                                if pm.character_bibles:
                                                    _p = apply_character_bibles(_p, pm.character_bibles)
                                                if _style_data:
                                                    _p = _style_data["prompt"].replace("{prompt}", _p)
                                                if _director and _director != "None":
                                                    _eff = _director
                                                    if _director == "Custom":
                                                        _eff = _s.get("custom_director", "")
                                                    if _eff:
                                                        _p += f". This video was directed by {_eff}."
                                                return _p

                                            if _type == "Vocal" and _vm == "Use Singer/Band Description":
                                                _src = _s.get("performance_desc", "")
                                                _assembled_vocal = _rh_assemble(_src) if _src else ""
                                                if not _assembled_vocal:
                                                    print(f"⚠️ [Run-ahead] Skipping {_sid}: performance_desc is empty")
                                                    return
                                                _base = _assembled_vocal
                                            else:
                                                _vp = _nr.iloc[0].get("Video_Prompt", "")
                                                _raw = "" if pd.isna(_vp) else str(_vp).strip()
                                                _base = _rh_assemble(_raw) if _raw else ""
                                            if not _base:
                                                print(f"⚠️ [Run-ahead] Skipping {_sid}: assembled prompt is empty (type={_type}, vocal_mode={_vm})")
                                                return

                                            # LLM conversion if requested
                                            if _rh_ni_copy.get('use_llm_image_prompt'):
                                                if _cmode == 'Regenerate both on each render':
                                                    # Always regenerate fresh — skip cache reads and writes
                                                    print(f"🧠 [Run-ahead] Converting prompt (no cache) for {_sid}...")
                                                    _zimage_prompt = convert_prompt_for_zimage(_base, pm, _s)
                                                else:
                                                    # Check CSV cache first
                                                    _rh_cached_p = ""
                                                    if "First_Frame_Prompt" in pm.df.columns:
                                                        _rh_raw = _nr.iloc[0].get("First_Frame_Prompt", "")
                                                        if _rh_raw and not pd.isna(_rh_raw) and str(_rh_raw).strip():
                                                            _rh_cached_p = str(_rh_raw).strip()
                                                    if _rh_cached_p:
                                                        _zimage_prompt = _rh_cached_p
                                                    elif _type == "Vocal" and _vm == "Use Singer/Band Description":
                                                        if (_s.get("zimage_vocal_source_assembled") == _base
                                                                and _s.get("zimage_vocal_first_frame_prompt")):
                                                            _zimage_prompt = _s["zimage_vocal_first_frame_prompt"]
                                                        else:
                                                            print(f"🧠 [Run-ahead] Converting prompt for {_sid}...")
                                                            _zimage_prompt = convert_prompt_for_zimage(_base, pm, _s)
                                                            pm.save_project_settings({
                                                                "zimage_vocal_first_frame_prompt": _zimage_prompt,
                                                                "zimage_vocal_source_assembled": _base,
                                                            })
                                                    else:
                                                        print(f"🧠 [Run-ahead] Converting prompt for {_sid}...")
                                                        _zimage_prompt = convert_prompt_for_zimage(_base, pm, _s)
                                                    # Cache converted prompt to CSV
                                                    _ridx = pm.df.index[pm.df['Shot_ID'] == _sid].tolist()
                                                    if _ridx and "First_Frame_Prompt" in pm.df.columns and not _rh_cached_p:
                                                        with pm.queue_lock:
                                                            pm.df.at[_ridx[0], "First_Frame_Prompt"] = _zimage_prompt
                                                            pm.save_data()
                                            else:
                                                _zimage_prompt = _base

                                            # Generate Z-image via ComfyUI (drain the generator synchronously)
                                            print(f"🚀 [Run-ahead] Generating Z-Image for {_sid}...")
                                            _frame_path, _frame_err = None, "Unknown"
                                            for _rh_item in generate_comfyui_zimage_first_frame(_zimage_prompt, _sid, pm):
                                                if isinstance(_rh_item, tuple):
                                                    _frame_path, _frame_err = _rh_item
                                                # progress strings discarded in background thread

                                            if _frame_err:
                                                print(f"❌ [Run-ahead] Z-Image failed for {_sid}: {_frame_err}")
                                                pm._runhead_result = None
                                            else:
                                                pm._runhead_result = (_sid, _frame_path, _zimage_prompt)
                                                print(f"✅ [Run-ahead] Z-Image ready for {_sid}: {_frame_path}")
                                        finally:
                                            pm._runhead_running = False

                                    _rh_t = threading.Thread(target=_runhead_worker, daemon=True)
                                    pm._runhead_thread = _rh_t
                                    _runhead_started_for_this_shot = True
                                    _rh_t.start()
                                    print(f"🚀 [Run-ahead] Started Z-Image pre-generation for {_rh_ni['shot_id']}")

                actual_render_secs = time.time() - render_start
                queue_elapsed += actual_render_secs
                config.record_render_time(
                    current_item['resolution'],
                    current_item.get('generation_mode', 'LTX-Native'),
                    shot_dur,
                    actual_render_secs
                )

                # VRAM leakage heuristic: check VRAM fullness + render slowdown
                _vram = config.get_vram_usage()
                _slowdown_ratio = actual_render_secs / max(shot_est, 1)
                _vram_warning = _vram and (_vram[0] / max(_vram[1], 1)) > config.VRAM_WARN_THRESHOLD
                _slowdown_warning = _slowdown_ratio > config.SLOWDOWN_WARN_FACTOR
                if _first_shot_in_session and _slowdown_ratio > 3.0:
                    pm.ltx_ram_warning = (
                        f"⚠️ FIRST SHOT ALERT: Render took {_slowdown_ratio:.1f}x estimated time. "
                        "Possible RAM leakage before queue started — consider restarting LTX Desktop now."
                    )
                elif _vram_warning or _slowdown_warning:
                    _warn_parts = []
                    if _vram_warning:
                        _warn_parts.append(f"GPU VRAM {_vram[0]:.1f}/{_vram[1]:.1f} GB ({100*_vram[0]/_vram[1]:.0f}% full)")
                    if _slowdown_warning:
                        _warn_parts.append(f"render took {_slowdown_ratio:.1f}x estimated time")
                    pm.ltx_ram_warning = "⚠️ Possible LTX RAM leakage: " + "; ".join(_warn_parts) + ". Consider restarting LTX Desktop."
                else:
                    pm.ltx_ram_warning = ""
                _first_shot_in_session = False

                sync_video_directory(pm)
                gal = get_project_videos(pm, proj)
                _shot_vids = sorted(
                    glob.glob(os.path.join(pm.get_path("videos"), f"{current_item['shot_id']}_*.mp4")),
                    key=os.path.getmtime
                )
                _new_vid = _shot_vids[-1] if _shot_vids else None
                yield (gal, format_queue_status(pm), [item[0] for item in gal], 0, 0, "", "", "", "",
                       _new_vid, _new_vid or gr.update(), gr.update())
        finally:
            with pm.queue_lock:
                pm.queue_processor_running = False
                pm.stop_video_generation = False
            # Clean up any stale run-ahead state
            pm._runhead_running = False
            pm._runhead_result = None
            pm._runhead_thread = None
            pm._runhead_shot_id = None
            gal = get_project_videos(pm, proj)
            yield gal, "💤 Queue is empty.", [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()

    def batch_enqueue_shots(mode, target_versions, resolution, vocal_mode, style, director, generation_mode, llm_image_prompt_mode, caching_mode, h3_prompt_cache_mode, vocal_chain_mode, zimage_backend, lora_path, pm):
        if pm.current_project:
            csv_path = os.path.join(pm.base_dir, pm.current_project, "shot_list.csv")
            if os.path.exists(csv_path):
                pm.df = pd.read_csv(csv_path)
                for col in config.REQUIRED_COLUMNS:
                    if col not in pm.df.columns:
                        pm.df[col] = ""
        df = pm.df
        if df.empty:
            return "No shots found.", gr.update(value="Start Batch Generation")

        if mode == "Regenerate all Shots":
            vid_dir = pm.get_path("videos")
            if os.path.exists(vid_dir):
                for f in glob.glob(os.path.join(vid_dir, "*.mp4")):
                    try: os.remove(f)
                    except: pass
            sync_video_directory(pm)
            current_gallery = []
            shot_ids = df['Shot_ID'].tolist()
        else:
            current_gallery = get_project_videos(pm)
            if mode in ("Generate all Action Shots", "Generate Remaining Action Shots"):
                shot_ids = df[df['Type'] == 'Action']['Shot_ID'].tolist()
            elif mode in ("Generate all Vocal Shots", "Generate Remaining Vocal Shots"):
                shot_ids = df[df['Type'] == 'Vocal']['Shot_ID'].tolist()
            else:
                shot_ids = df['Shot_ID'].tolist()

        items_added = 0
        downgraded_count = 0
        for shot_id in shot_ids:
            row = df[df['Shot_ID'] == shot_id]
            if row.empty:
                continue
            if pd.isna(row.iloc[0].get('Video_Prompt')) or not str(row.iloc[0].get('Video_Prompt')).strip():
                continue
            effective_res = _effective_resolution(shot_id, resolution, df)
            if effective_res != resolution:
                downgraded_count += 1
            current_count = get_video_count_for_shot(shot_id, current_gallery)
            needed = max(0, target_versions - current_count)
            for _ in range(needed):
                item = {'shot_id': shot_id, 'resolution': effective_res,
                        'vocal_mode': vocal_mode, 'style': style,
                        'director': director, 'generation_mode': generation_mode,
                        'delete_path': None, 'camera_motion': 'none',
                        'use_llm_image_prompt': (llm_image_prompt_mode == "Convert with LLM"),
                        'caching_mode': caching_mode,
                        'h3_prompt_cache_mode': h3_prompt_cache_mode,
                        'vocal_chain_mode': vocal_chain_mode,
                        'zimage_backend': zimage_backend,
                        'lora_path': lora_path}
                with pm.queue_lock:
                    pm.render_queue.append(item)
                items_added += 1

        if items_added == 0:
            return "ℹ️ No shots need generation.\n" + format_queue_status(pm), gr.update(value="Start Batch Generation")
        msg = f"✅ Added {items_added} item(s) to queue."
        if downgraded_count:
            msg += f" ⚠️ {downgraded_count} shot(s) downgraded to 720p (duration >5s)."
        return msg + "\n" + format_queue_status(pm), gr.update(value=f"⏳ Queue: {items_added} items")

    single_shot_btn.click(
        lambda shot_id, res, vocal, style, director, gen_mode, cam, llm_img, caching_mode, h3_cache_mode, vocal_chain, zimage_backend, lora, pm:
            add_to_render_queue(shot_id, res, vocal, style, director, gen_mode, pm,
                                camera_motion=cam, use_llm_image_prompt=(llm_img == "Convert with LLM"),
                                caching_mode=caching_mode, h3_prompt_cache_mode=h3_cache_mode,
                                vocal_chain_mode=vocal_chain,
                                zimage_backend=zimage_backend, lora_path=lora),
        inputs=[single_shot_dropdown, vid_resolution_dropdown, vid_vocal_prompt_mode,
                vid_style_dropdown, vid_director_dropdown, vid_firstframe_mode,
                single_shot_camera_dropdown, llm_image_prompt_dropdown, first_frame_reuse_dropdown,
                h3_prompt_cache_dropdown,
                vid_vocal_chain_checkbox, vid_zimage_backend, vid_lora_dropdown, pm_state],
        outputs=[vid_gen_status]
    ).then(
        process_render_queue_if_idle,
        inputs=[pm_state, current_proj_var],
        outputs=[vid_gallery, vid_gen_status, gallery_paths_state,
                 current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt,
                 vid_large_view, selected_vid_path, first_frame_prompt_edit],
        show_progress="hidden"
    )

    def toggle_queue_pause(pm):
        runtime = getattr(pm, "pipeline_runtime", None)
        if runtime and runtime.is_active():
            pm.queue_paused = not pm.queue_paused
            runtime.set_paused(pm.queue_paused)
        else:
            pm.queue_paused = not pm.queue_paused
        return "▶ Resume Queue" if pm.queue_paused else "⏸ Pause Queue"

    queue_pause_btn.click(toggle_queue_pause, inputs=[pm_state], outputs=[queue_pause_btn])

    def cancel_pending_queue(pm):
        with pm.queue_lock:
            count = len(pm.render_queue)
            pm.render_queue.clear()
        runtime = getattr(pm, "pipeline_runtime", None)
        if runtime and runtime.is_active():
            count += runtime.cancel_pending()
        return f"🧹 Cancelled {count} pending item(s); active work will finish.", gr.update()

    def stop_render_queue(pm):
        with pm.queue_lock:
            pm.render_queue.clear()
            pm.stop_video_generation = True
            pm.queue_paused = False
        runtime = getattr(pm, "pipeline_runtime", None)
        if runtime and runtime.is_active():
            runtime.stop()
            return "⏹ Stopping owned H3 jobs on both ComfyUI instances...", "⏸ Pause Queue"
        return "🚫 Stopping current render and clearing the queue...", "⏸ Pause Queue"

    queue_cancel_btn.click(
        cancel_pending_queue, inputs=[pm_state], outputs=[vid_gen_status, queue_pause_btn]
    ).then(
        lambda: (0, 0, "", "", "", ""),
        outputs=[current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt]
    )

    vid_gen_start_btn.click(
        batch_enqueue_shots,
        inputs=[vid_gen_mode_dropdown, vid_versions_dropdown, vid_resolution_dropdown,
                vid_vocal_prompt_mode, vid_style_dropdown, vid_director_dropdown, vid_firstframe_mode,
                llm_image_prompt_dropdown, first_frame_reuse_dropdown, h3_prompt_cache_dropdown,
                vid_vocal_chain_checkbox,
                vid_zimage_backend, vid_lora_dropdown, pm_state],
        outputs=[vid_gen_status, vid_gen_start_btn]
    ).then(
        process_render_queue_if_idle,
        inputs=[pm_state, current_proj_var],
        outputs=[vid_gallery, vid_gen_status, gallery_paths_state,
                 current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt,
                 vid_large_view, selected_vid_path, first_frame_prompt_edit],
        show_progress="hidden"
    ).then(
        lambda: gr.update(value="Start Batch Generation"),
        outputs=[vid_gen_start_btn]
    )

    vid_gen_stop_btn.click(
        stop_render_queue, inputs=[pm_state], outputs=[vid_gen_status, queue_pause_btn]
    ).then(
        lambda: (0, 0, "", "", "", ""),
        outputs=[current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt]
    )

    def handle_vid_delete(path_to_del, proj, pm, gallery_paths):
        try:
            current_idx = gallery_paths.index(path_to_del)
        except ValueError:
            current_idx = -1

        new_gal, _ = delete_video_file(path_to_del, proj, pm)
        new_paths = [item[0] for item in new_gal]

        next_path = ""
        next_shot_id = ""
        if new_paths and current_idx >= 0:
            next_idx = min(current_idx, len(new_paths) - 1)
            next_path = new_paths[next_idx]
            fname = os.path.basename(next_path)
            next_shot_id = fname.split('_')[0] if '_' in fname else ""

        return new_gal, next_path or None, next_path, new_paths

    del_vid_btn.click(handle_vid_delete, inputs=[selected_vid_path, current_proj_var, pm_state, gallery_paths_state], outputs=[vid_gallery, vid_large_view, selected_vid_path, gallery_paths_state])

    def handle_regen_vid_and_prompt(shot_id_txt, selected_path, resolution, vocal_mode, style, director, generation_mode, llm_image_prompt_mode, caching_mode, h3_prompt_cache_mode, vocal_chain_mode, camera_motion, lora_path, proj, pm):
        use_llm_img = (llm_image_prompt_mode == "Convert with LLM")
        if not shot_id_txt:
            yield gr.update(), "❌ No Shot ID selected", gr.update(), 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
            return
        if pm.llm_busy:
            yield gr.update(), "⚠️ LLM already running — please wait.", gr.update(), 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
            return
        pm.llm_busy = True
        try:
            settings = pm.load_project_settings()
            llm_model = config.LM_STUDIO_MODEL
            plot = settings.get("plot", "")
            prompt_template = settings.get("prompt_template", config.DEFAULT_CONCEPT_PROMPT)
            performance_desc = settings.get("performance_desc", "")

            gal = get_project_videos(pm, proj)
            yield gal, f"⏳ Generating new prompt for {shot_id_txt}...", [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
            time.sleep(0.1)

            llm = LLMBridge()
            row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id_txt).upper()].tolist()
            if not row_idx:
                gal = get_project_videos(pm, proj)
                yield gal, f"❌ Shot {shot_id_txt} not found.", [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
                return
            index = row_idx[0]
            row = pm.df.loc[index]

            if row['Type'] == 'Vocal':
                final_vid_prompt = performance_desc
            else:
                loc_pos = pm.df.index.get_loc(index)
                if loc_pos > 0:
                    prev_index = pm.df.index[loc_pos - 1]
                    prev_shot_text = pm.df.loc[prev_index, 'Video_Prompt']
                    if pd.isna(prev_shot_text): prev_shot_text = "N/A"
                else:
                    prev_shot_text = "None (Start of video)"

                filled_prompt = prompt_template.replace("{plot}", plot)\
                    .replace("{type}", row['Type'])\
                    .replace("{start}", f"{row['Start_Time']:.1f}")\
                    .replace("{duration}", f"{row['Duration']:.1f}")\
                    .replace("{prev_shot}", prev_shot_text)

                final_vid_prompt = llm.query(config.LTX_SYSTEM_PROMPT, filled_prompt, llm_model)

            pm.df.at[index, 'Video_Prompt'] = final_vid_prompt
            # Clear stale first-frame caches since the video prompt changed
            if "First_Frame_Prompt" in pm.df.columns:
                pm.df.at[index, 'First_Frame_Prompt'] = ""
            for column in ("First_Frame_Prompt_Purpose", "First_Frame_Prompt_Source_Hash"):
                if column in pm.df.columns:
                    pm.df.at[index, column] = ""
            for column in ("First_Frame_Image_Path", "First_Frame_Image_Source",
                           "First_Frame_Image_Aspect", "First_Frame_Image_Prompt_Hash"):
                if column in pm.df.columns:
                    pm.df.at[index, column] = ""
            # Clear override — new LLM prompt replaces the old assembled version
            if 'Prompt_Override' in pm.df.columns:
                pm.df.at[index, 'Prompt_Override'] = ''
            if 'Prompt_Override_Text' in pm.df.columns:
                pm.df.at[index, 'Prompt_Override_Text'] = ''
            pm.save_data()

            # If Z-Image + LLM conversion active: regenerate first-frame prompt now so it's visible in the UI
            new_ffp = ""
            if generation_mode in ("Krea 2 First Frame", "Z-Image First Frame") and use_llm_img:
                gal = get_project_videos(pm, proj)
                yield gal, f"🧠 Regenerating first-frame image prompt...", [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), gr.update()
                if row['Type'] == 'Vocal' and vocal_mode == "Use Singer/Band Description":
                    base_for_image = performance_desc
                else:
                    base_for_image = final_vid_prompt
                new_ffp = convert_prompt_for_zimage(base_for_image, pm, settings)
                if "First_Frame_Prompt" in pm.df.columns:
                    pm.df.at[index, 'First_Frame_Prompt'] = new_ffp
                    if "First_Frame_Prompt_Purpose" in pm.df.columns:
                        pm.df.at[index, "First_Frame_Prompt_Purpose"] = (
                            "h3_vocal_storyboard"
                            if config.VIDEO_BACKEND == "MiniMax H3" and row['Type'] == 'Vocal'
                            and vocal_mode == "Use Storyboard Prompt"
                            else "zimage_generated"
                        )
                    if "First_Frame_Prompt_Source_Hash" in pm.df.columns:
                        pm.df.at[index, "First_Frame_Prompt_Source_Hash"] = hashlib.sha256(
                            base_for_image.encode("utf-8")
                        ).hexdigest()
                    pm.save_data()

            add_to_render_queue(shot_id_txt, resolution, vocal_mode, style, director, generation_mode, pm,
                                delete_path=selected_path, camera_motion=camera_motion,
                                use_llm_image_prompt=use_llm_img,
                                caching_mode=caching_mode,
                                h3_prompt_cache_mode=h3_prompt_cache_mode,
                                vocal_chain_mode=vocal_chain_mode,
                                lora_path=lora_path)
            gal = get_project_videos(pm, proj)
            yield gal, f"✅ Prompt saved. Added to queue.\n" + format_queue_status(pm), [item[0] for item in gal], 0, 0, "", "", "", "", gr.update(), gr.update(), new_ffp
        finally:
            pm.llm_busy = False

    regen_vid_same_prompt_btn.click(
        lambda shot_id, sel_path, res, vocal, style, director, gen_mode, llm_img, caching_mode, h3_cache_mode, vocal_chain, cam, lora, pm:
            add_to_render_queue(shot_id, res, vocal, style, director, gen_mode, pm,
                                delete_path=sel_path, camera_motion=cam,
                                use_llm_image_prompt=(llm_img == "Convert with LLM"),
                                caching_mode=caching_mode, h3_prompt_cache_mode=h3_cache_mode,
                                vocal_chain_mode=vocal_chain,
                                lora_path=lora),
        inputs=[single_shot_dropdown, selected_vid_path, vid_resolution_dropdown,
                vid_vocal_prompt_mode, vid_style_dropdown, vid_director_dropdown, vid_firstframe_mode,
                llm_image_prompt_dropdown, first_frame_reuse_dropdown, h3_prompt_cache_dropdown,
                vid_vocal_chain_checkbox,
                single_shot_camera_dropdown, vid_lora_dropdown, pm_state],
        outputs=[vid_gen_status]
    ).then(
        process_render_queue_if_idle,
        inputs=[pm_state, current_proj_var],
        outputs=[vid_gallery, vid_gen_status, gallery_paths_state,
                 current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt,
                 vid_large_view, selected_vid_path, first_frame_prompt_edit],
        show_progress="hidden"
    )

    regen_vid_new_prompt_btn.click(
        handle_regen_vid_and_prompt,
        inputs=[single_shot_dropdown, selected_vid_path, vid_resolution_dropdown,
                vid_vocal_prompt_mode, vid_style_dropdown, vid_director_dropdown, vid_firstframe_mode,
                llm_image_prompt_dropdown, first_frame_reuse_dropdown, h3_prompt_cache_dropdown,
                vid_vocal_chain_checkbox,
                single_shot_camera_dropdown, vid_lora_dropdown, current_proj_var, pm_state],
        outputs=[vid_gallery, vid_gen_status, gallery_paths_state,
                 current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt,
                 vid_large_view, selected_vid_path,
                 first_frame_prompt_edit],
        show_progress="hidden"
    ).then(
        process_render_queue_if_idle,
        inputs=[pm_state, current_proj_var],
        outputs=[vid_gallery, vid_gen_status, gallery_paths_state,
                 current_render_progress, queue_progress_bar,
                 current_render_eta, queue_eta_txt,
                 current_render_cost, queue_cost_txt,
                 vid_large_view, selected_vid_path, first_frame_prompt_edit],
        show_progress="hidden"
    )

    def handle_regen_first_frame_prompt(shot_id, vocal_mode, style, director, pm):
        if not shot_id or pm.df.empty:
            gr.Warning("No shot selected.")
            return gr.update()
        if pm.llm_busy:
            gr.Warning("LLM already running — please wait.")
            return gr.update()
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            gr.Warning(f"Shot {shot_id} not found.")
            return gr.update()
        pm.llm_busy = True
        try:
            settings = pm.load_project_settings()
            row = pm.df.loc[row_idx[0]]
            if row.get("Type") == "Vocal" and vocal_mode == "Use Singer/Band Description":
                base = settings.get("performance_desc", "")
            else:
                vp = row.get("Video_Prompt", "")
                base = "" if pd.isna(vp) else str(vp).strip()
            if not base:
                gr.Warning("No base prompt to convert (Video Prompt is empty).")
                return gr.update()
            # Apply character bibles, style, and director — same assembly as generate_video_for_shot
            if pm.character_bibles:
                base = apply_character_bibles(base, pm.character_bibles)
            style_data = resolve_style_data(style, pm) if style and style != "None" else None
            if style_data:
                base = style_data["prompt"].replace("{prompt}", base)
            if director and director != "None":
                eff_director = director
                if director == "Custom":
                    eff_director = settings.get("custom_director", "")
                if eff_director:
                    base += f". This video was directed by {eff_director}."
            new_ffp = convert_prompt_for_zimage(base, pm, settings)
            if "First_Frame_Prompt" in pm.df.columns:
                pm.df.at[row_idx[0], "First_Frame_Prompt"] = new_ffp
                if "First_Frame_Prompt_Purpose" in pm.df.columns:
                    purpose = (
                        "h3_vocal_storyboard"
                        if config.VIDEO_BACKEND == "MiniMax H3" and row.get("Type") == "Vocal"
                        and vocal_mode == "Use Storyboard Prompt"
                        else "h3_action_target" if config.VIDEO_BACKEND == "MiniMax H3"
                        else "zimage_generated"
                    )
                    pm.df.at[row_idx[0], "First_Frame_Prompt_Purpose"] = purpose
                if "First_Frame_Prompt_Source_Hash" in pm.df.columns:
                    pm.df.at[row_idx[0], "First_Frame_Prompt_Source_Hash"] = hashlib.sha256(
                        base.encode("utf-8")
                    ).hexdigest()
            # Clear cached first frame image — new prompt means a new image should be generated
            for column in ("First_Frame_Image_Path", "First_Frame_Image_Source",
                           "First_Frame_Image_Aspect", "First_Frame_Image_Prompt_Hash"):
                if column in pm.df.columns:
                    pm.df.at[row_idx[0], column] = ""
            pm.save_data()
            gr.Info(f"First-frame prompt regenerated for {shot_id}.")
            return new_ffp
        finally:
            pm.llm_busy = False

    regen_first_frame_prompt_btn.click(
        handle_regen_first_frame_prompt,
        inputs=[single_shot_dropdown, vid_vocal_prompt_mode, vid_style_dropdown, vid_director_dropdown, pm_state],
        outputs=[first_frame_prompt_edit]
    ).then(
        get_first_frame_img_status,
        inputs=[single_shot_dropdown, pm_state],
        outputs=[first_frame_img_status]
    )

    tab3_ui.select(update_single_shot_choices, inputs=[pm_state, shared_shot_state], outputs=[single_shot_dropdown, shared_shot_state])

    return {
        "tab3_ui": tab3_ui,
        "backend_state": backend_state,
        "vid_resolution_dropdown": vid_resolution_dropdown,
        "single_shot_dropdown": single_shot_dropdown,
        "single_shot_type_radio": single_shot_type_radio,
        "vid_gallery": vid_gallery,
        "gallery_paths_state": gallery_paths_state,
        "vid_gen_start_btn": vid_gen_start_btn,
        "current_render_progress": current_render_progress,
        "queue_progress_bar": queue_progress_bar,
        "current_render_eta": current_render_eta,
        "queue_eta_txt": queue_eta_txt,
        "current_render_cost": current_render_cost,
        "queue_cost_txt": queue_cost_txt,
        # Preference controls restored on project load
        "vid_firstframe_mode": vid_firstframe_mode,
        "vid_vocal_chain_checkbox": vid_vocal_chain_checkbox,
        "vid_zimage_backend": vid_zimage_backend,
        "llm_image_prompt_dropdown": llm_image_prompt_dropdown,
        "first_frame_reuse_dropdown": first_frame_reuse_dropdown,
        "h3_prompt_cache_row": h3_prompt_cache_row,
        "h3_prompt_cache_dropdown": h3_prompt_cache_dropdown,
        "h3_prompt_cache_status": h3_prompt_cache_status,
        "first_frame_prompt_row": first_frame_prompt_row,
        "first_frame_img_status": first_frame_img_status,
        "vid_vocal_prompt_mode": vid_vocal_prompt_mode,
        "vid_gen_mode_dropdown": vid_gen_mode_dropdown,
        "vid_versions_dropdown": vid_versions_dropdown,
        "single_shot_camera_dropdown": single_shot_camera_dropdown,
        "vid_director_dropdown": vid_director_dropdown,
        "vid_style_dropdown": vid_style_dropdown,
        "vid_lora_dropdown": vid_lora_dropdown,
        "lora_row": lora_row,
        "h3_row": h3_row,
        "h3_aspect_dropdown": h3_aspect_dropdown,
        "h3_quality_dropdown": h3_quality_dropdown,
        "h3_lipsync_output_dropdown": h3_lipsync_output_dropdown,
        "h3_custom_width": h3_custom_width,
        "h3_custom_height": h3_custom_height,
        "override_status": override_status,
        "clear_override_btn": clear_override_btn,
    }
