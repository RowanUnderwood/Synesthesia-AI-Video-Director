import os

import gradio as gr
import pandas as pd
from pydub import AudioSegment

import config
from timeline import get_existing_projects, scan_vocals_advanced, build_simple_timeline
from llm_logic import (generate_overarching_plot, generate_performance_description,
                       generate_concepts_logic, generate_character_bibles_logic,
                       stop_gen, generate_story_file, generate_all_firstframe_prompts_logic)
from h3 import generate_h3_character_references, h3_reference_gallery as get_h3_reference_gallery, h3_reference_paths
from utils import get_file_path


def h3_reference_ui_updates(pm, preferred=None):
    """Return synchronized H3 character controls for the project's current bible state."""
    names = list(pm.character_bibles) if pm and pm.current_project else []
    settings = pm.load_project_settings() if pm and pm.current_project else {}
    lead = settings.get("h3_lead_character", "")
    selected = preferred if preferred in names else (lead if lead in names else (names[0] if names else None))
    face, body = h3_reference_paths(pm, selected) if selected else (None, None)
    return (
        gr.update(choices=names, value=lead if lead in names else None),
        gr.update(choices=names, value=selected),
        face, body, get_h3_reference_gallery(pm) if pm and pm.current_project else [],
    )


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
                min_shot_dur = gr.Slider(1, 15, value=2, label="Min Duration (s)")
                max_shot_dur = gr.Slider(1, 15, value=4, label="Max Duration (s)")
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
                    "Use **Add Character** for manual entries, then click **Save Changes** after editing the table."
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
                    add_bible_character_btn = gr.Button("➕ Add Character")
                    save_bible_changes_btn = gr.Button("💾 Save Changes", variant="primary")

            with gr.Accordion("🧬 MiniMax H3 Character References", open=False):
                gr.Markdown(
                    "MiniMax H3 uses one face closeup and one full-body/wardrobe reference for each "
                    "character. Generate references here before rendering H3 shots. Ref2VA uses the generated "
                    "setting/target frame as Picture 1, then accepts up to four named bible characters per shot "
                    "as Pictures 2–9 (two identity images per character)."
                )
                with gr.Row():
                    h3_lead_character = gr.Dropdown(
                        choices=[], label="Lead Singer for H3 Lip-Sync", info="Vocal H3 shots use this character's two references."
                    )
                    h3_reference_character = gr.Dropdown(
                        choices=[], label="Character Reference to Generate / View"
                    )
                    h3_generate_character_btn = gr.Button("Generate / Regenerate Character Pair", variant="primary")
                    h3_generate_all_btn = gr.Button("Generate All Character Pairs")
                with gr.Row():
                    h3_face_preview = gr.Image(label="Face Closeup", interactive=False, type="filepath")
                    h3_body_preview = gr.Image(label="Full-Body Portrait", interactive=False, type="filepath")
                h3_reference_status = gr.Textbox(label="H3 Reference Status", interactive=False)
                h3_reference_gallery = gr.Gallery(label="Generated H3 Character References", columns=4, height=280, allow_preview=True)

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

    t2_inputs = [current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, rough_concept_in, plot_out, performance_desc_in, video_mode_drp, scripted_total_dur, scripted_shot_count, pm_state,
                 singer_gender_in, ffp_style_dropdown, ffp_director_dropdown]

    def auto_save_tab2(proj_name, min_sil, sil_thresh, mode, min_d, max_d, concept, plot, performance_d, video_mode, s_total_dur, s_shot_count, pm,
                       singer_gender, ffp_style, ffp_director):
        if proj_name:
            pm.current_project = proj_name
            settings = {
                "min_silence": min_sil, "silence_thresh": sil_thresh, "shot_mode": mode,
                "min_dur": min_d, "max_dur": max_d,
                "rough_concept": concept, "plot": plot,
                "performance_desc": performance_d,
                "video_mode": video_mode,
                "scripted_total_dur": s_total_dur, "scripted_shot_count": s_shot_count,
                "singer_gender": singer_gender,
                "last_ffp_style": ffp_style,
                "last_ffp_director": ffp_director,
            }
            pm.save_project_settings(settings)

    for tab2_comp in [min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, video_mode_drp, scripted_total_dur, scripted_shot_count,
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

    stop_concepts_btn.click(stop_gen, inputs=[pm_state], outputs=[concept_gen_status])

    def _add_bible_character(value):
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
        elif isinstance(value, list):
            frame = pd.DataFrame(value, columns=["character_name", "description"])
        else:
            frame = pd.DataFrame(columns=["character_name", "description"])
        for column in ("character_name", "description"):
            if column not in frame.columns:
                frame[column] = ""
        frame = pd.concat(
            [frame[["character_name", "description"]], pd.DataFrame([{
                "character_name": "", "description": "",
            }])],
            ignore_index=True,
        )
        return frame, "✏️ Blank character added. Enter a name and description, then click Save Changes."

    add_bible_character_btn.click(
        _add_bible_character,
        inputs=[bible_table],
        outputs=[bible_table, bible_status],
    )

    def _save_bible_changes(value, current_reference, pm):
        if not pm or not pm.current_project:
            refs = h3_reference_ui_updates(pm)
            empty = pd.DataFrame(columns=["character_name", "description"])
            shots = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
            return ("❌ Load a project before saving character bibles.", empty, shots) + refs

        old_bibles = dict(pm.character_bibles)
        old_references = {name for name in old_bibles if any(h3_reference_paths(pm, name))}
        try:
            bible_df = pm.replace_character_bibles(value)
        except Exception as exc:
            current_df = value if isinstance(value, pd.DataFrame) else pd.DataFrame(
                value or [], columns=["character_name", "description"]
            )
            return (f"❌ {exc}", current_df, pm.df) + h3_reference_ui_updates(pm, current_reference)

        names = list(pm.character_bibles)
        settings = pm.load_project_settings()
        if settings.get("h3_lead_character") and settings["h3_lead_character"] not in names:
            pm.save_project_settings({"h3_lead_character": ""})
        old_keys = {name.casefold() for name in old_bibles}
        added = [name for name in names if name.casefold() not in old_keys]
        preferred = added[-1] if added else current_reference
        stale = [
            name for name in names
            if name in old_references and old_bibles.get(name) != pm.character_bibles.get(name)
        ]
        status = f"✅ Saved {len(names)} character bible entr{'y' if len(names) == 1 else 'ies'}."
        if stale:
            status += f" Regenerate stale H3 references for: {', '.join(stale)}."
        return (status, bible_df, pm.df) + h3_reference_ui_updates(pm, preferred)

    save_bible_changes_btn.click(
        _save_bible_changes,
        inputs=[bible_table, h3_reference_character, pm_state],
        outputs=[
            bible_status, bible_table, shot_table,
            h3_lead_character, h3_reference_character, h3_face_preview, h3_body_preview,
            h3_reference_gallery,
        ],
    )

    def _save_h3_lead(name, pm):
        if pm and pm.current_project:
            pm.save_project_settings({"h3_lead_character": name or ""})

    h3_lead_character.change(_save_h3_lead, inputs=[h3_lead_character, pm_state])

    def _show_h3_reference(name, pm):
        face, body = h3_reference_paths(pm, name) if name else (None, None)
        return face, body

    h3_reference_character.change(
        _show_h3_reference,
        inputs=[h3_reference_character, pm_state],
        outputs=[h3_face_preview, h3_body_preview],
    )

    def _generate_h3_references(name, pm):
        if getattr(pm, "queue_processor_running", False) or (
            getattr(pm, "pipeline_runtime", None) and pm.pipeline_runtime.is_active()
        ):
            return "❌ Character references are unavailable while the render queue is running.", None, None, get_h3_reference_gallery(pm)
        if getattr(pm, "character_reference_busy", False):
            return "⚠️ Character-reference generation is already running.", None, None, get_h3_reference_gallery(pm)
        pm.character_reference_busy = True
        try:
            names = [name] if name else []
            completed = generate_h3_character_references(pm, names)
            face, body = h3_reference_paths(pm, name) if name else (None, None)
            return f"✅ Generated H3 references for: {', '.join(completed)}", face, body, get_h3_reference_gallery(pm)
        except Exception as exc:
            return f"❌ {exc}", None, None, get_h3_reference_gallery(pm) if pm and pm.current_project else []
        finally:
            pm.character_reference_busy = False

    def _generate_all_h3_references(pm):
        if getattr(pm, "queue_processor_running", False) or (
            getattr(pm, "pipeline_runtime", None) and pm.pipeline_runtime.is_active()
        ):
            return "❌ Character references are unavailable while the render queue is running.", get_h3_reference_gallery(pm)
        if getattr(pm, "character_reference_busy", False):
            return "⚠️ Character-reference generation is already running.", get_h3_reference_gallery(pm)
        pm.character_reference_busy = True
        try:
            completed = generate_h3_character_references(pm)
            return f"✅ Generated H3 references for: {', '.join(completed)}", get_h3_reference_gallery(pm)
        except Exception as exc:
            return f"❌ {exc}", get_h3_reference_gallery(pm) if pm and pm.current_project else []
        finally:
            pm.character_reference_busy = False

    h3_generate_character_btn.click(
        _generate_h3_references,
        inputs=[h3_reference_character, pm_state],
        outputs=[h3_reference_status, h3_face_preview, h3_body_preview, h3_reference_gallery],
    )
    h3_generate_all_btn.click(
        _generate_all_h3_references,
        inputs=[pm_state],
        outputs=[h3_reference_status, h3_reference_gallery],
    )

    export_csv_btn.click(lambda pm: pm.export_csv(), inputs=[pm_state], outputs=csv_downloader)
    import_csv_btn.upload(lambda f, pm: pm.import_csv(f), inputs=[import_csv_btn, pm_state], outputs=[import_status, shot_table]).then(lambda: gr.update(value=None), outputs=[import_csv_btn])
    download_story_btn.click(generate_story_file, inputs=[pm_state], outputs=[story_downloader])
    export_bibles_btn.click(lambda pm: pm.export_character_bibles(), inputs=[pm_state], outputs=bibles_downloader)
    import_bibles_btn.upload(
        lambda f, pm: pm.import_character_bibles(f),
        inputs=[import_bibles_btn, pm_state],
        outputs=[import_bibles_status, bible_table],
    ).then(
        lambda pm: (pm.df,) + h3_reference_ui_updates(pm),
        inputs=[pm_state],
        outputs=[
            shot_table, h3_lead_character, h3_reference_character, h3_face_preview,
            h3_body_preview, h3_reference_gallery,
        ],
    ).then(lambda: gr.update(value=None), outputs=[import_bibles_btn])

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

    def refresh_tab2(pm):
        refs = h3_reference_ui_updates(pm)
        return (pm.df,) + refs

    tab2_ui.select(
        refresh_tab2,
        inputs=[pm_state],
        outputs=[shot_table, h3_lead_character, h3_reference_character, h3_face_preview, h3_body_preview, h3_reference_gallery],
    )

    return {
        "tab2_ui": tab2_ui,
        "shot_table": shot_table,
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
        "h3_lead_character": h3_lead_character,
        "h3_reference_character": h3_reference_character,
        "h3_face_preview": h3_face_preview,
        "h3_body_preview": h3_body_preview,
        "h3_reference_gallery": h3_reference_gallery,
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
    h3_lead_character = t2["h3_lead_character"]
    h3_reference_character = t2["h3_reference_character"]
    h3_face_preview = t2["h3_face_preview"]
    h3_body_preview = t2["h3_body_preview"]
    h3_reference_gallery = t2["h3_reference_gallery"]
    rough_concept_in = t2["rough_concept_in"]
    plot_out = t2["plot_out"]
    performance_desc_in = t2["performance_desc_in"]
    singer_gender_in = t2["singer_gender_in"]
    llm_dropdown = t5["llm_model_dropdown"]
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
    ).then(
        h3_reference_ui_updates,
        inputs=[pm_state],
        outputs=[h3_lead_character, h3_reference_character, h3_face_preview, h3_body_preview, h3_reference_gallery],
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
    ).then(
        h3_reference_ui_updates,
        inputs=[pm_state],
        outputs=[h3_lead_character, h3_reference_character, h3_face_preview, h3_body_preview, h3_reference_gallery],
    )
