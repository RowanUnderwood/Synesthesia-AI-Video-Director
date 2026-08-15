import os

import gradio as gr
import pandas as pd
from pydub import AudioSegment

import config
from models import LLMBridge
from timeline import get_existing_projects, scan_vocals_advanced, build_simple_timeline
from llm_logic import (generate_overarching_plot, generate_performance_description,
                       generate_concepts_logic, generate_character_bibles_logic,
                       stop_gen, generate_story_file, generate_all_firstframe_prompts_logic)
from utils import get_file_path


def build(pm_state, current_proj_var, shared_shot_state, vocals_up, lyrics_in):
    """Build Tab 2: Storyboard. Returns dict of exported components."""

    with gr.Tab("2. Storyboard") as tab2_ui:
        with gr.Accordion("Step 1: Timeline Settings", open=True):
            with gr.Row():
                video_mode_drp = gr.Dropdown(["Intercut", "All Vocals", "All Action", "Scripted"], value="Intercut", label="Mode")
            with gr.Row():
                min_silence_sl = gr.Slider(500, 2000, value=700, label="Min Silence (ms)")
                silence_thresh_sl = gr.Slider(-60, -20, value=-45, label="Silence Threshold (dB)")
            with gr.Row():
                shot_mode_drp = gr.Dropdown(["Fixed", "Random"], value="Random", label="Shot Duration Mode")
                min_shot_dur = gr.Slider(1, 10, value=2, label="Min Duration (s)")
                max_shot_dur = gr.Slider(1, 10, value=4, label="Max Duration (s)")
            with gr.Row():
                gr.Markdown("ℹ️ Shots over 5 seconds require 720p or lower resolution. 1080p selections will automatically downgrade to 720p for these shots.")
            with gr.Row(visible=False) as scripted_duration_row:
                scripted_total_dur = gr.Number(label="Total Duration (seconds)", value=60, precision=0)
                scripted_shot_count = gr.Number(label="Number of Shots (alternative)", value=0, precision=0)
                gr.Markdown("*Specify total duration OR shot count. If both > 0, total duration takes priority.*")
            with gr.Row():
                scan_btn = gr.Button("1. Scan Vocals & Build Timeline", variant="primary")
                scan_status = gr.Textbox(label="Build Status", interactive=False)

        with gr.Accordion("Step 2: Plot & Concept Generation", open=True):
            with gr.Row():
                avail_models = LLMBridge().get_models()
                last_model = config.get_global_llm()
                if not last_model:
                    last_model = avail_models[0] if avail_models else "qwen3-vl-8b-instruct-abliterated-v2.0"

                llm_dropdown = gr.Dropdown(choices=avail_models, value=last_model, label="Select LLM Model", interactive=True, allow_custom_value=True)
                refresh_llm_btn = gr.Button("🔄", size="sm")

                llm_dropdown.change(config.save_global_llm, inputs=[llm_dropdown])

            with gr.Row():
                rough_concept_in = gr.Textbox(label="Rough User Concept / Vibe (Optional)", placeholder="e.g. A cyberpunk rainstorm...", scale=2, lines=5)
                with gr.Column(scale=1):
                    singer_gender_in = gr.Textbox(label="Singer Gender (Optional)", placeholder="e.g. Female, Male, Non-binary (Leave blank to invent)", lines=1)
                    gen_performance_btn = gr.Button("Generate Singer, Band & Venue Desc")
                    performance_desc_in = gr.Textbox(label="Singer, Band, and Venue Description (Also used as Prompt for Vocal Shots)", placeholder="Short description of the singer, band, and venue setup", lines=2)

            gen_plot_btn = gr.Button("2. Generate Overarching Plot")
            plot_out = gr.Textbox(label="Overarching Plot (Optional)", lines=4, interactive=True)

            with gr.Row():
                gen_concepts_btn = gr.Button("3. Generate Video Prompts (Bulk Generation)", variant="primary")
                stop_concepts_btn = gr.Button("Stop Generation", variant="stop")

            concept_gen_status = gr.Textbox(label="Concept Generation Status", interactive=False)

            with gr.Row():
                gen_firstframe_prompts_btn = gr.Button("3b. Generate All First Frame Prompts (Z-Image)", variant="secondary")
            with gr.Row():
                ffp_style_dropdown = gr.Dropdown(choices=config.STYLE_NAMES, value="None", label="Style (for First Frame Prompts)")
                ffp_director_dropdown = gr.Dropdown(choices=config.DIRECTORS, value="None", label="Directed by (for First Frame Prompts)")
            gen_firstframe_status = gr.Textbox(label="First Frame Prompt Status", interactive=False, visible=False)

            with gr.Accordion("📖 Character Bibles", open=False):
                gr.Markdown(
                    "After generating video prompts, click **Generate Character Bibles** to have the LLM "
                    "identify all recurring named characters and build a visual description for each. "
                    "These descriptions are automatically injected into each LTX video prompt at generation time. "
                    "Edit the table below to refine descriptions — changes auto-save."
                )
                gen_bible_btn = gr.Button("Generate Character Bibles")
                bible_status = gr.Textbox(label="Bible Generation Status", interactive=False)
                bible_table = gr.Dataframe(
                    headers=["character_name", "description"],
                    label="Character Bibles (Editable)",
                    interactive=True,
                    wrap=True,
                    type="pandas"
                )

        with gr.Row():
            gr.Markdown("### 📂 Data Management")
            with gr.Row():
                export_csv_btn = gr.Button("Export CSV")
                csv_downloader = gr.File(label="Download Shot List", interactive=False)
            with gr.Row():
                download_story_btn = gr.Button("Download Story (.txt)")
                story_downloader = gr.File(label="Story Text File", interactive=False)
            with gr.Row():
                import_csv_btn = gr.UploadButton("Import CSV (Update Prompts)", file_types=[".csv"])
                import_status = gr.Textbox(label="Import Status", interactive=False)
            with gr.Row():
                export_bibles_btn = gr.Button("Export Bibles CSV")
                bibles_downloader = gr.File(label="Character Bibles CSV", interactive=False)
            with gr.Row():
                import_bibles_btn = gr.UploadButton("Import Bibles CSV", file_types=[".csv"])
                import_bibles_status = gr.Textbox(label="Bible Import Status", interactive=False)

        shot_table = gr.Dataframe(headers=config.REQUIRED_COLUMNS, interactive=True, wrap=True, type="pandas")

    # --- Tab 2 Internal Events ---

    t2_inputs = [current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, rough_concept_in, plot_out, performance_desc_in, video_mode_drp, scripted_total_dur, scripted_shot_count, pm_state,
                 singer_gender_in, ffp_style_dropdown, ffp_director_dropdown]

    def auto_save_tab2(proj_name, min_sil, sil_thresh, mode, min_d, max_d, llm, concept, plot, performance_d, video_mode, s_total_dur, s_shot_count, pm,
                       singer_gender, ffp_style, ffp_director):
        if proj_name:
            pm.current_project = proj_name
            settings = {
                "min_silence": min_sil, "silence_thresh": sil_thresh, "shot_mode": mode,
                "min_dur": min_d, "max_dur": max_d, "llm_model": llm,
                "rough_concept": concept, "plot": plot,
                "performance_desc": performance_d,
                "video_mode": video_mode,
                "scripted_total_dur": s_total_dur, "scripted_shot_count": s_shot_count,
                "singer_gender": singer_gender,
                "last_ffp_style": ffp_style,
                "last_ffp_director": ffp_director,
            }
            pm.save_project_settings(settings)

    for tab2_comp in [min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, video_mode_drp, scripted_total_dur, scripted_shot_count,
                      ffp_style_dropdown, ffp_director_dropdown]:
        tab2_comp.change(auto_save_tab2, inputs=t2_inputs)

    for tab2_text_comp in [rough_concept_in, plot_out, performance_desc_in, singer_gender_in]:
        tab2_text_comp.blur(auto_save_tab2, inputs=t2_inputs)

    def on_mode_change(mode):
        is_scripted = (mode == "Scripted")
        is_intercut = (mode == "Intercut")

        silence_vis = gr.update(visible=is_intercut)
        scripted_vis = gr.update(visible=is_scripted)

        if is_scripted:
            scan_label = gr.update(value="1. Build Timeline")
        else:
            scan_label = gr.update(value="1. Scan Vocals & Build Timeline") if is_intercut else gr.update(value="1. Build Timeline")

        if is_scripted:
            gender_label = gr.update(label="Main Character's Gender (Optional)")
            perf_label = gr.update(label="Main Character and Setting Description")
            perf_btn_label = gr.update(value="Generate Main Character & Setting Desc")
        else:
            gender_label = gr.update(label="Singer Gender (Optional)")
            perf_label = gr.update(label="Singer, Band, and Venue Description (Also used as Prompt for Vocal Shots)")
            perf_btn_label = gr.update(value="Generate Singer, Band & Venue Desc")

        return [silence_vis, silence_vis, scripted_vis, scan_label, gender_label, perf_label, perf_btn_label]

    video_mode_drp.change(
        on_mode_change,
        inputs=[video_mode_drp],
        outputs=[min_silence_sl, silence_thresh_sl, scripted_duration_row, scan_btn, singer_gender_in, performance_desc_in, gen_performance_btn]
    )

    def run_scan(v_file, p_name, m_sil, s_thr, s_mode, min_d, max_d, v_mode, s_total_dur, s_shot_count, pm):
        yield "⏳ Initializing...", pm.df
        if not p_name:
            yield "❌ Error: No project selected.", pm.df
            return
        pm.current_project = p_name

        if v_mode == "Intercut":
            final_v_path = get_file_path(v_file) or pm.get_asset_path_if_exists("vocals.mp3")
            if not final_v_path or not os.path.exists(final_v_path):
                yield "❌ Error: No vocals file found.", pm.df
                return
            yield "⏳ Detecting silence and building timeline (this may take a moment)...", pm.df
            df = scan_vocals_advanced(final_v_path, p_name, m_sil, s_thr, s_mode, min_d, max_d, pm)

        elif v_mode in ("All Vocals", "All Action"):
            audio_path = get_file_path(v_file) or pm.get_asset_path_if_exists("vocals.mp3") or pm.get_asset_path_if_exists("full_song.mp3")
            if not audio_path or not os.path.exists(audio_path):
                yield "❌ Error: No audio file found. Upload a vocals or full song file.", pm.df
                return
            try:
                audio = AudioSegment.from_file(audio_path)
                total_dur = audio.duration_seconds
            except Exception as e:
                yield f"❌ Error loading audio: {e}", pm.df
                return
            shot_type = "Vocal" if v_mode == "All Vocals" else "Action"
            yield f"⏳ Building {v_mode.lower()} timeline ({total_dur:.1f}s)...", pm.df
            df = build_simple_timeline(total_dur, shot_type, s_mode, min_d, max_d, pm)

        elif v_mode == "Scripted":
            total_dur = 0
            if s_total_dur and s_total_dur > 0:
                total_dur = float(s_total_dur)
            elif s_shot_count and s_shot_count > 0:
                avg_dur = (min_d + max_d) / 2.0
                total_dur = float(s_shot_count) * avg_dur
            else:
                yield "❌ Error: Specify a Total Duration or Number of Shots for Scripted mode.", pm.df
                return
            yield f"⏳ Building scripted timeline ({total_dur:.1f}s)...", pm.df
            df = build_simple_timeline(total_dur, "Action", s_mode, min_d, max_d, pm)
        else:
            yield "❌ Error: Unknown mode.", pm.df
            return

        if df.empty:
            yield "❌ Error: Could not build timeline. Check settings.", pm.df
        else:
            yield "✅ Timeline Built Successfully!", df

    scan_btn.click(run_scan, inputs=[vocals_up, current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, video_mode_drp, scripted_total_dur, scripted_shot_count, pm_state], outputs=[scan_status, shot_table])

    refresh_llm_btn.click(lambda: gr.update(choices=LLMBridge().get_models()), outputs=llm_dropdown)

    stop_concepts_btn.click(stop_gen, inputs=[pm_state], outputs=[concept_gen_status])

    def save_bible_edits(new_df, pm):
        if pm.current_project and new_df is not None:
            try:
                bibles = {}
                for _, row in new_df.iterrows():
                    name = str(row.get("character_name", "")).strip()
                    desc = str(row.get("description", "")).strip()
                    if name and name.lower() != "nan":
                        bibles[name] = desc
                pm.character_bibles = bibles
                pm.save_character_bibles()
                pm.update_characters_column()
                pm.save_data()
            except Exception as e:
                print(f"Error saving bible edits: {e}")

    bible_table.change(save_bible_edits, inputs=[bible_table, pm_state])

    export_csv_btn.click(lambda pm: pm.export_csv(), inputs=[pm_state], outputs=csv_downloader)
    import_csv_btn.upload(lambda f, pm: pm.import_csv(f), inputs=[import_csv_btn, pm_state], outputs=[import_status, shot_table]).then(lambda: gr.update(value=None), outputs=[import_csv_btn])
    download_story_btn.click(generate_story_file, inputs=[pm_state], outputs=[story_downloader])
    export_bibles_btn.click(lambda pm: pm.export_character_bibles(), inputs=[pm_state], outputs=bibles_downloader)
    import_bibles_btn.upload(lambda f, pm: pm.import_character_bibles(f), inputs=[import_bibles_btn, pm_state], outputs=[import_bibles_status, bible_table]).then(lambda: gr.update(value=None), outputs=[import_bibles_btn])

    def save_manual_df_edits(new_df, pm):
        if pm.current_project:
            if isinstance(new_df, list):
                if new_df and len(new_df[0]) == len(config.REQUIRED_COLUMNS):
                    new_df = pd.DataFrame(new_df, columns=config.REQUIRED_COLUMNS)
                else:
                    return
            pm.df = new_df
            pm.save_data()

    shot_table.change(save_manual_df_edits, inputs=[shot_table, pm_state])

    tab2_ui.select(lambda pm: pm.df, inputs=[pm_state], outputs=[shot_table])

    return {
        "tab2_ui": tab2_ui,
        "shot_table": shot_table,
        "llm_dropdown": llm_dropdown,
        "video_mode_drp": video_mode_drp,
        # All fields needed as handle_load outputs
        "min_silence_sl": min_silence_sl,
        "silence_thresh_sl": silence_thresh_sl,
        "shot_mode_drp": shot_mode_drp,
        "min_shot_dur": min_shot_dur,
        "max_shot_dur": max_shot_dur,
        "rough_concept_in": rough_concept_in,
        "plot_out": plot_out,
        "performance_desc_in": performance_desc_in,
        "scripted_total_dur": scripted_total_dur,
        "scripted_shot_count": scripted_shot_count,
        "scripted_duration_row": scripted_duration_row,
        "scan_btn": scan_btn,
        "singer_gender_in": singer_gender_in,
        "gen_performance_btn": gen_performance_btn,
        "gen_plot_btn": gen_plot_btn,
        "gen_concepts_btn": gen_concepts_btn,
        "gen_firstframe_prompts_btn": gen_firstframe_prompts_btn,
        "gen_bible_btn": gen_bible_btn,
        "concept_gen_status": concept_gen_status,
        "gen_firstframe_status": gen_firstframe_status,
        "bible_table": bible_table,
        "bible_status": bible_status,
        "ffp_style_dropdown": ffp_style_dropdown,
        "ffp_director_dropdown": ffp_director_dropdown,
    }


def wire_template_events(t2, t5, pm_state, vocals_up, lyrics_in):
    """Wire Tab 2 generation button events that depend on template components from Tab 5.
    Must be called from app.py after both tab2 and tab5 have been built."""
    gen_performance_btn = t2["gen_performance_btn"]
    gen_plot_btn = t2["gen_plot_btn"]
    gen_concepts_btn = t2["gen_concepts_btn"]
    gen_firstframe_prompts_btn = t2["gen_firstframe_prompts_btn"]
    gen_bible_btn = t2["gen_bible_btn"]
    concept_gen_status = t2["concept_gen_status"]
    gen_firstframe_status = t2["gen_firstframe_status"]
    bible_status = t2["bible_status"]
    bible_table = t2["bible_table"]
    shot_table = t2["shot_table"]
    rough_concept_in = t2["rough_concept_in"]
    plot_out = t2["plot_out"]
    performance_desc_in = t2["performance_desc_in"]
    singer_gender_in = t2["singer_gender_in"]
    llm_dropdown = t2["llm_dropdown"]
    video_mode_drp = t2["video_mode_drp"]
    ffp_style_dropdown = t2["ffp_style_dropdown"]
    ffp_director_dropdown = t2["ffp_director_dropdown"]

    plot_sys_prompt_in = t5["plot_sys_prompt_in"]
    plot_user_template_in = t5["plot_user_template_in"]
    plot_sys_prompt_scripted_in = t5["plot_sys_prompt_scripted_in"]
    plot_user_template_scripted_in = t5["plot_user_template_scripted_in"]
    perf_sys_prompt_in = t5["perf_sys_prompt_in"]
    perf_user_template_in = t5["perf_user_template_in"]
    perf_sys_prompt_scripted_in = t5["perf_sys_prompt_scripted_in"]
    perf_user_template_scripted_in = t5["perf_user_template_scripted_in"]
    concepts_bulk_template_in = t5["concepts_bulk_template_in"]
    concepts_vocals_template_in = t5["concepts_vocals_template_in"]
    concepts_scripted_template_in = t5["concepts_scripted_template_in"]
    bible_sys_prompt_in = t5["bible_sys_prompt_in"]
    bible_user_template_in = t5["bible_user_template_in"]
    zimage_template_in = t5["zimage_template_in"]

    gen_performance_btn.click(
        generate_performance_description,
        inputs=[rough_concept_in, plot_out, singer_gender_in, llm_dropdown, video_mode_drp,
                perf_sys_prompt_in, perf_user_template_in, perf_sys_prompt_scripted_in, perf_user_template_scripted_in],
        outputs=performance_desc_in
    )
    gen_plot_btn.click(
        generate_overarching_plot,
        inputs=[rough_concept_in, lyrics_in, llm_dropdown, pm_state, video_mode_drp,
                plot_sys_prompt_in, plot_user_template_in, plot_sys_prompt_scripted_in, plot_user_template_scripted_in],
        outputs=plot_out
    )
    gen_concepts_btn.click(
        generate_concepts_logic,
        inputs=[plot_out, llm_dropdown, rough_concept_in, performance_desc_in, pm_state,
                video_mode_drp, singer_gender_in, concepts_bulk_template_in,
                concepts_vocals_template_in, concepts_scripted_template_in],
        outputs=[shot_table, concept_gen_status]
    ).then(
        generate_character_bibles_logic,
        inputs=[pm_state, llm_dropdown, video_mode_drp, bible_sys_prompt_in, bible_user_template_in],
        outputs=[bible_status, bible_table, shot_table]
    )
    gen_firstframe_prompts_btn.click(
        lambda: gr.update(visible=True),
        outputs=[gen_firstframe_status]
    ).then(
        generate_all_firstframe_prompts_logic,
        inputs=[pm_state, llm_dropdown, zimage_template_in, ffp_style_dropdown, ffp_director_dropdown],
        outputs=[gen_firstframe_status]
    )
    gen_bible_btn.click(
        generate_character_bibles_logic,
        inputs=[pm_state, llm_dropdown, video_mode_drp, bible_sys_prompt_in, bible_user_template_in],
        outputs=[bible_status, bible_table, shot_table]
    )
