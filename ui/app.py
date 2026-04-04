import os

import gradio as gr
import pandas as pd

import config
from models import ProjectManager
from timeline import get_existing_projects
from video import get_project_videos
from utils import header_html, format_time, get_file_path

from . import tab1_project as tab1
from . import tab2_storyboard as tab2
from . import tab3_video as tab3
from . import tab4_assembly as tab4
from . import tab5_settings as tab5
from . import tab6_help as tab6

css = """
.header-row {
    align-items: center !important;
    gap: 12px !important;
    padding: 0 !important;
    margin-bottom: 0 !important;
}
.header-row > div {
    flex-grow: 0 !important;
}
.header-row > div:last-child {
    flex-grow: 1 !important;
}
"""


def build_app():
    with gr.Blocks(title="Synesthesia AI Video Director", theme=gr.themes.Default(), css=css) as app:
        pm_state = gr.State(ProjectManager())
        shared_shot_state = gr.State(None)
        current_proj_var = gr.State("")

        with gr.Row():
            gr.HTML(header_html)

        with gr.Tabs():
            t1 = tab1.build(pm_state, current_proj_var)
            t2 = tab2.build(pm_state, current_proj_var, shared_shot_state,
                            vocals_up=t1["vocals_up"], lyrics_in=t1["lyrics_in"])
            t3 = tab3.build(pm_state, current_proj_var, shared_shot_state)
            tab4.build(pm_state, shared_shot_state, current_proj_var,
                       shot_table=t2["shot_table"],
                       song_up=t1["song_up"],
                       vid_resolution_dropdown=t3["vid_resolution_dropdown"],
                       vid_gallery=t3["vid_gallery"],
                       gallery_paths_state=t3["gallery_paths_state"])
            t5 = tab5.build(pm_state, llm_dropdown=t2["llm_dropdown"])
            tab6.build()

        # Wire Tab 2 generation events that depend on template components in Tab 5
        tab2.wire_template_events(t2, t5, pm_state, t1["vocals_up"], t1["lyrics_in"])

        # ==========================================
        # BACKEND CHANGE → CAP MAX DURATION SLIDER
        def on_backend_change_cap(backend):
            max_val = 3 if backend == "Wan2GP" else 5
            return gr.update(maximum=max_val)

        t5["video_backend_drp"].change(
            on_backend_change_cap,
            inputs=[t5["video_backend_drp"]],
            outputs=[t2["max_shot_dur"]],
        )

        # ==========================================
        # CROSS-TAB EVENT WIRING
        # handle_create and handle_load span Tab 1 + 2 + 3 outputs simultaneously
        # ==========================================

        def handle_create(name, pm):
            msg = pm.create_project(name)
            clean_name = pm.sanitize_name(name)

            _blank_update = gr.update()
            if "already exists" in msg or "Invalid" in msg:
                return (msg,) + (_blank_update,) * 26

            settings = pm.load_project_settings()
            df = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
            bible_df = pd.DataFrame(columns=["character_name", "description"])

            return (
                msg,
                gr.update(choices=get_existing_projects(), value=clean_name),
                clean_name,
                "00h00m00s",
                df,
                "",   # rough_concept_in
                "",   # plot_out
                "",   # performance_desc_in
                [],   # vid_gallery
                bible_df,
                None, None, "",  # vocals_up, song_up, lyrics_in
                settings.get("plot_sys_prompt_music", config.DEFAULT_PLOT_SYSTEM_PROMPT_MUSIC),
                settings.get("plot_user_template_music", config.DEFAULT_PLOT_USER_TEMPLATE_MUSIC),
                settings.get("plot_sys_prompt_scripted", config.DEFAULT_PLOT_SYSTEM_PROMPT_SCRIPTED),
                settings.get("plot_user_template_scripted", config.DEFAULT_PLOT_USER_TEMPLATE_SCRIPTED),
                settings.get("perf_sys_prompt_music", config.DEFAULT_PERF_SYSTEM_PROMPT_MUSIC),
                settings.get("perf_user_template_music", config.DEFAULT_PERF_USER_TEMPLATE_MUSIC),
                settings.get("perf_sys_prompt_scripted", config.DEFAULT_PERF_SYSTEM_PROMPT_SCRIPTED),
                settings.get("perf_user_template_scripted", config.DEFAULT_PERF_USER_TEMPLATE_SCRIPTED),
                settings.get("concepts_bulk_template", config.BULK_PROMPT_TEMPLATE),
                settings.get("concepts_vocals_template", config.ALL_VOCALS_PROMPT_TEMPLATE),
                settings.get("concepts_scripted_template", config.SCRIPTED_PROMPT_TEMPLATE),
                settings.get("bible_sys_prompt", config.CHARACTER_BIBLE_SYSTEM_PROMPT),
                settings.get("bible_user_template", config.CHARACTER_BIBLE_USER_TEMPLATE),
                settings.get("zimage_prompt_template", config.DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE),
            )

        t1["create_btn"].click(
            handle_create,
            inputs=[t1["proj_name"], pm_state],
            outputs=[t1["proj_status"], t1["project_dropdown"], current_proj_var, t1["time_spent_disp"],
                     t2["shot_table"], t2["rough_concept_in"], t2["plot_out"], t2["performance_desc_in"],
                     t3["vid_gallery"], t2["bible_table"],
                     t1["vocals_up"], t1["song_up"], t1["lyrics_in"],
                     t5["plot_sys_prompt_in"], t5["plot_user_template_in"],
                     t5["plot_sys_prompt_scripted_in"], t5["plot_user_template_scripted_in"],
                     t5["perf_sys_prompt_in"], t5["perf_user_template_in"],
                     t5["perf_sys_prompt_scripted_in"], t5["perf_user_template_scripted_in"],
                     t5["concepts_bulk_template_in"], t5["concepts_vocals_template_in"], t5["concepts_scripted_template_in"],
                     t5["bible_sys_prompt_in"], t5["bible_user_template_in"],
                     t5["zimage_template_in"]]
        )

        def handle_load(name, pm):
            msg, df = pm.load_project(name)
            lyrics = pm.get_lyrics()
            v_path = pm.get_asset_path_if_exists("vocals.mp3")
            s_path = pm.get_asset_path_if_exists("full_song.mp3")
            settings = pm.load_project_settings()

            gal_vids = get_project_videos(pm, name)
            gal_paths = [item[0] for item in gal_vids]
            time_str = format_time(pm.total_time_spent)

            loaded_mode = settings.get("video_mode", "Intercut")
            is_scripted = (loaded_mode == "Scripted")
            is_intercut = (loaded_mode == "Intercut")

            bible_df = pd.DataFrame(
                list(pm.character_bibles.items()), columns=["character_name", "description"]
            ) if pm.character_bibles else pd.DataFrame(columns=["character_name", "description"])

            return (
                msg, time_str, df, lyrics, v_path, s_path,
                gr.update(value=settings.get("min_silence", 700), visible=is_intercut),
                gr.update(value=settings.get("silence_thresh", -45), visible=is_intercut),
                settings.get("shot_mode", "Random"), settings.get("min_dur", 2), settings.get("max_dur", 4),
                settings.get("llm_model", "qwen3-vl-8b-instruct-abliterated-v2.0"), settings.get("rough_concept", ""),
                settings.get("plot", ""),
                settings.get("prompt_template", config.DEFAULT_CONCEPT_PROMPT),
                gr.update(value=settings.get("performance_desc", ""), label="Main Character and Setting Description" if is_scripted else "Singer, Band, and Venue Description (Also used as Prompt for Vocal Shots)"),
                name,
                gal_vids, gal_paths, gr.update(value="Start Batch Generation", variant="primary"),
                loaded_mode,
                settings.get("scripted_total_dur", 60),
                settings.get("scripted_shot_count", 0),
                gr.update(visible=is_scripted),
                gr.update(value="1. Build Timeline" if not is_intercut else "1. Scan Vocals & Build Timeline"),
                gr.update(label="Main Character's Gender (Optional)" if is_scripted else "Singer Gender (Optional)",
                          value=settings.get("singer_gender", "")),
                gr.update(value="Generate Main Character & Setting Desc" if is_scripted else "Generate Singer, Band & Venue Desc"),
                settings.get("plot_sys_prompt_music", config.DEFAULT_PLOT_SYSTEM_PROMPT_MUSIC),
                settings.get("plot_user_template_music", config.DEFAULT_PLOT_USER_TEMPLATE_MUSIC),
                settings.get("plot_sys_prompt_scripted", config.DEFAULT_PLOT_SYSTEM_PROMPT_SCRIPTED),
                settings.get("plot_user_template_scripted", config.DEFAULT_PLOT_USER_TEMPLATE_SCRIPTED),
                settings.get("perf_sys_prompt_music", config.DEFAULT_PERF_SYSTEM_PROMPT_MUSIC),
                settings.get("perf_user_template_music", config.DEFAULT_PERF_USER_TEMPLATE_MUSIC),
                settings.get("perf_sys_prompt_scripted", config.DEFAULT_PERF_SYSTEM_PROMPT_SCRIPTED),
                settings.get("perf_user_template_scripted", config.DEFAULT_PERF_USER_TEMPLATE_SCRIPTED),
                settings.get("concepts_bulk_template", config.BULK_PROMPT_TEMPLATE),
                settings.get("concepts_vocals_template", config.ALL_VOCALS_PROMPT_TEMPLATE),
                settings.get("concepts_scripted_template", config.SCRIPTED_PROMPT_TEMPLATE),
                settings.get("bible_sys_prompt", config.CHARACTER_BIBLE_SYSTEM_PROMPT),
                settings.get("bible_user_template", config.CHARACTER_BIBLE_USER_TEMPLATE),
                bible_df,
                settings.get("zimage_prompt_template", config.DEFAULT_ZIMAGE_PROMPT_CONVERSION_TEMPLATE),
                # Tab 2 FFP preferences
                settings.get("last_ffp_style", "None"),
                settings.get("last_ffp_director", "None"),
                # Tab 3 generation preferences
                settings.get("firstframe_mode", "LTX-Native") if settings.get("firstframe_mode", "LTX-Native") in ("LTX-Native", "Z-Image First Frame") else "LTX-Native",
                gr.update(visible=(settings.get("firstframe_mode", "LTX-Native") == "Z-Image First Frame"),
                          value=settings.get("llm_image_prompt_mode", "Use video prompt as-is")),
                gr.update(visible=(settings.get("firstframe_mode", "LTX-Native") == "Z-Image First Frame"),
                          value=settings.get("first_frame_reuse_mode", "Use cached prompt")),
                gr.update(visible=(settings.get("firstframe_mode", "LTX-Native") == "Z-Image First Frame")),
                gr.update(visible=(settings.get("firstframe_mode", "LTX-Native") == "Z-Image First Frame"),
                          value=settings.get("zimage_backend", "LTX Desktop")),
                settings.get("vocal_prompt_mode", "Use Singer/Band Description"),
                settings.get("last_gen_mode", "Generate Remaining Shots"),
                settings.get("last_versions", 1),
                settings.get("last_resolution", "1080p"),
                settings.get("last_camera_motion", "none"),
                settings.get("last_director", "None"),
                settings.get("last_style", "None"),
                settings.get("vocal_chain_mode", False),
            )

        t1["load_btn"].click(
            handle_load,
            inputs=[t1["project_dropdown"], pm_state],
            outputs=[
                t1["proj_status"], t1["time_spent_disp"], t2["shot_table"], t1["lyrics_in"], t1["vocals_up"], t1["song_up"],
                t2["min_silence_sl"], t2["silence_thresh_sl"], t2["shot_mode_drp"], t2["min_shot_dur"], t2["max_shot_dur"],
                t2["llm_dropdown"], t2["rough_concept_in"], t2["plot_out"], t5["prompt_template_in"],
                t2["performance_desc_in"],
                current_proj_var,
                t3["vid_gallery"], t3["gallery_paths_state"], t3["vid_gen_start_btn"],
                t2["video_mode_drp"], t2["scripted_total_dur"], t2["scripted_shot_count"],
                t2["scripted_duration_row"], t2["scan_btn"],
                t2["singer_gender_in"], t2["gen_performance_btn"],
                t5["plot_sys_prompt_in"], t5["plot_user_template_in"],
                t5["plot_sys_prompt_scripted_in"], t5["plot_user_template_scripted_in"],
                t5["perf_sys_prompt_in"], t5["perf_user_template_in"],
                t5["perf_sys_prompt_scripted_in"], t5["perf_user_template_scripted_in"],
                t5["concepts_bulk_template_in"], t5["concepts_vocals_template_in"], t5["concepts_scripted_template_in"],
                t5["bible_sys_prompt_in"], t5["bible_user_template_in"],
                t2["bible_table"],
                t5["zimage_template_in"],
                # Tab 2 FFP preferences
                t2["ffp_style_dropdown"], t2["ffp_director_dropdown"],
                # Tab 3 generation preferences
                t3["vid_firstframe_mode"], t3["llm_image_prompt_dropdown"],
                t3["first_frame_reuse_dropdown"], t3["first_frame_prompt_row"],
                t3["vid_zimage_backend"],
                t3["vid_vocal_prompt_mode"], t3["vid_gen_mode_dropdown"],
                t3["vid_versions_dropdown"], t3["vid_resolution_dropdown"],
                t3["single_shot_camera_dropdown"], t3["vid_director_dropdown"],
                t3["vid_style_dropdown"],
                t3["vid_vocal_chain_checkbox"],
            ]
        )

    return app
