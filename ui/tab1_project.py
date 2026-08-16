import os
import shutil

import gradio as gr
import pandas as pd

import config
from external_llm_export import ExternalExportError, create_external_llm_bundle
from lyric_timing import (LyricTimingError, align_caption_cues, align_lyric_lines,
                          alignment_needs_fallback, better_alignment,
                          parse_timed_captions, preferred_fallback_model,
                          save_alignment, transcribe_audio)
from timeline import get_existing_projects
from utils import get_file_path, format_time


def build(pm_state, current_proj_var):
    """Build Tab 1: Project & Assets. Returns dict of components needed by other tabs."""

    with gr.Tab("1. Project & Assets") as tab1_ui:
        gr.Markdown("### Create or Load")
        with gr.Row():
            with gr.Column():
                proj_name = gr.Textbox(label="New Project Name", placeholder="MyMusicVideo_v1")
                create_btn = gr.Button("Create New Project")
            with gr.Column():
                with gr.Row():
                    project_dropdown = gr.Dropdown(choices=get_existing_projects(), label="Select Existing Project", interactive=True)
                    refresh_proj_btn = gr.Button("🔄", size="sm")
                with gr.Row():
                    load_btn = gr.Button("Load Selected Project")
                    delete_proj_btn = gr.Button("Delete Selected Project", variant="stop")
                with gr.Row(visible=False) as confirm_delete_row:
                    gr.Markdown("⚠️ **Are you sure?** This permanently deletes the project and all its files.")
                    confirm_delete_btn = gr.Button("Yes, Delete It", variant="stop")
                    cancel_delete_btn = gr.Button("Cancel")

        with gr.Row():
            proj_status = gr.Textbox(label="System Status", interactive=False)
            time_spent_disp = gr.Textbox(label="Total Project Time", interactive=False)

        gr.Markdown("### Assets")
        with gr.Row():
            vocals_up = gr.Audio(label="Upload Vocals (Audio)", type="filepath")
            song_up = gr.Audio(label="Upload Full Song (Audio)", type="filepath")
            lyrics_in = gr.Textbox(label="Lyrics", lines=5)
        with gr.Row():
            add_lyric_timestamps_btn = gr.Button("Add Lyric Timestamps")
            export_external_llm_btn = gr.Button("Export Templates and Instructions for External LLM")
        with gr.Row():
            timed_caption_file = gr.File(
                label="Timed Captions (SBV, SRT, VTT, or LRC)",
                file_types=[".sbv", ".srt", ".vtt", ".lrc"],
                type="filepath",
            )
            import_timed_captions_btn = gr.Button("Import Timed Captions")
        with gr.Row():
            lyric_timestamp_status = gr.Textbox(label="Lyric Timestamp Status", interactive=False)
            external_export_status = gr.Textbox(label="External LLM Export Status", interactive=False)
        external_export_file = gr.File(label="External LLM Director Bundle", interactive=False)

    # --- Tab 1 Internal Events ---

    refresh_proj_btn.click(lambda: gr.update(choices=get_existing_projects()), outputs=[project_dropdown])

    def handle_delete_project(name, pm):
        if not name: return "No project selected.", gr.update()
        path = os.path.join(pm.base_dir, name)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                if pm.current_project == name:
                    pm.current_project = None
                    pm.df = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
                return f"Deleted project '{name}'.", gr.update(choices=get_existing_projects(), value=None)
            except Exception as e:
                return f"Error deleting project: {e}", gr.update()
        return "Project not found.", gr.update()

    delete_proj_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[confirm_delete_row, delete_proj_btn]
    )

    confirm_delete_btn.click(
        handle_delete_project,
        inputs=[project_dropdown, pm_state],
        outputs=[proj_status, project_dropdown]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[confirm_delete_row, delete_proj_btn]
    )

    cancel_delete_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[confirm_delete_row, delete_proj_btn]
    )

    def auto_save_lyrics(proj_name_val, text, pm):
        if proj_name_val:
            pm.current_project = proj_name_val
            pm.save_lyrics(text)

    def auto_save_files(proj_name_val, v_file, s_file, pm):
        if proj_name_val:
            v_src = get_file_path(v_file)
            s_src = get_file_path(s_file)
            if v_src: pm.save_asset(v_src, "vocals.mp3")
            if s_src: pm.save_asset(s_src, "full_song.mp3")

    lyrics_in.change(auto_save_lyrics, inputs=[current_proj_var, lyrics_in, pm_state])

    for file_comp in [vocals_up, song_up]:
        file_comp.upload(auto_save_files, inputs=[current_proj_var, vocals_up, song_up, pm_state])
        file_comp.clear(auto_save_files, inputs=[current_proj_var, vocals_up, song_up, pm_state])

    def add_lyric_timestamps(lyrics_text, pm):
        if not pm or not pm.current_project:
            yield "❌ Load a project before adding lyric timestamps.", gr.update()
            return
        vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
        if not vocals_path or not os.path.isfile(vocals_path):
            yield "❌ Upload an isolated vocals track first.", gr.update()
            return
        if not str(lyrics_text or "").strip():
            yield "❌ Paste the ground-truth lyrics first.", gr.update()
            return
        try:
            yield "⏳ Loading small Whisper on CPU and scanning isolated vocals...", gr.update()
            transcription = transcribe_audio(vocals_path, "small")
            yield (
                f"⏳ Locally aligning {len(transcription['words'])} recognized words while checking "
                "line spans and repeated sections...",
                gr.update(),
            )
            selected = None
            alignment = None
            if transcription["words"]:
                alignment = align_lyric_lines(
                    lyrics_text, transcription["words"], transcription.get("duration")
                )
                selected = (alignment, transcription)
            fallback_warning = ""
            if alignment is None or alignment_needs_fallback(alignment):
                fallback_model = preferred_fallback_model(transcription.get("language", ""))
                small_result = (
                    f"{alignment['matched_lines']} of {alignment['eligible_lines']} lines"
                    if alignment is not None else "no words"
                )
                yield (
                    f"⏳ Small Whisper produced {small_result}. Retrying with CPU {fallback_model}; "
                    "its first use may download a larger model...",
                    gr.update(),
                )
                try:
                    fallback_transcription = transcribe_audio(vocals_path, fallback_model)
                    fallback_alignment = align_lyric_lines(
                        lyrics_text,
                        fallback_transcription["words"],
                        fallback_transcription.get("duration"),
                    )
                    fallback_result = (fallback_alignment, fallback_transcription)
                    selected = (
                        fallback_result if selected is None
                        else better_alignment(selected, fallback_result)
                    )
                except LyricTimingError as fallback_exc:
                    if selected is None:
                        raise LyricTimingError(
                            f"Small Whisper detected no usable words, and the stronger-model retry "
                            f"failed: {fallback_exc}"
                        ) from fallback_exc
                    fallback_warning = f" Stronger-model retry failed: {fallback_exc}"
            if selected is None:
                raise LyricTimingError("Whisper did not produce a usable lyric alignment.")
            alignment, transcription = selected
            if alignment["matched_lines"] < 1:
                raise LyricTimingError(
                    "No lyric lines matched confidently. The existing Lyrics field was left unchanged."
                )
            save_alignment(
                pm,
                vocals_path,
                alignment,
                transcription.get("language", "unknown"),
                transcription.get("model", "small"),
                transcription=transcription,
            )
            unmatched = alignment["eligible_lines"] - alignment["matched_lines"]
            status = (
                f"✅ Timestamped {alignment['matched_lines']} of {alignment['eligible_lines']} lyric lines "
                f"using CPU {transcription.get('model', 'Whisper')} "
                f"({transcription.get('language', 'unknown')})."
            )
            if unmatched:
                status += f" {unmatched} line(s) remain untimestamped because confidence was too low."
            status += fallback_warning
            yield status, alignment["timestamped_lyrics"]
        except Exception as exc:
            message = str(exc) if isinstance(exc, LyricTimingError) else f"Unexpected timestamping error: {exc}"
            yield f"❌ {message}", gr.update()

    add_lyric_timestamps_btn.click(
        add_lyric_timestamps,
        inputs=[lyrics_in, pm_state],
        outputs=[lyric_timestamp_status, lyrics_in],
    )

    def import_timed_captions(caption_file, lyrics_text, pm):
        if not pm or not pm.current_project:
            return "❌ Load a project before importing timed captions.", gr.update()
        caption_path = get_file_path(caption_file)
        if not caption_path or not os.path.isfile(caption_path):
            return "❌ Select an SBV, SRT, VTT, or LRC caption file first.", gr.update()
        vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
        if not vocals_path or not os.path.isfile(vocals_path):
            return "❌ Upload the corresponding isolated vocals track first.", gr.update()
        if not str(lyrics_text or "").strip():
            return "❌ Paste the ground-truth lyrics first.", gr.update()
        try:
            cues, caption_format = parse_timed_captions(caption_path)
            alignment = align_caption_cues(lyrics_text, cues)
            if alignment["matched_lines"] < 1:
                raise LyricTimingError(
                    "No caption cues matched the ground-truth lyric lines confidently."
                )
            save_alignment(
                pm,
                vocals_path,
                alignment,
                "from captions",
                "caption-import",
                source="caption_import",
                transcription={
                    "caption_format": caption_format,
                    "caption_filename": os.path.basename(caption_path),
                    "cues": cues,
                },
            )
            unmatched = alignment["eligible_lines"] - alignment["matched_lines"]
            status = (
                f"✅ Imported {caption_format} timing for {alignment['matched_lines']} of "
                f"{alignment['eligible_lines']} lyric lines."
            )
            if unmatched:
                status += f" {unmatched} line(s) did not match and remain untimestamped."
            return status, alignment["timestamped_lyrics"]
        except LyricTimingError as exc:
            return f"❌ {exc}", gr.update()
        except Exception as exc:
            return f"❌ Timed-caption import failed: {exc}", gr.update()

    import_timed_captions_btn.click(
        import_timed_captions,
        inputs=[timed_caption_file, lyrics_in, pm_state],
        outputs=[lyric_timestamp_status, lyrics_in],
    )

    def export_external_llm(pm):
        try:
            return create_external_llm_bundle(pm)
        except ExternalExportError as exc:
            return f"❌ {exc}", None
        except Exception as exc:
            return f"❌ External-LLM export failed: {exc}", None

    export_external_llm_btn.click(
        export_external_llm,
        inputs=[pm_state],
        outputs=[external_export_status, external_export_file],
    )

    return {
        "tab1_ui": tab1_ui,
        "proj_name": proj_name,
        "create_btn": create_btn,
        "project_dropdown": project_dropdown,
        "load_btn": load_btn,
        "proj_status": proj_status,
        "time_spent_disp": time_spent_disp,
        "vocals_up": vocals_up,
        "song_up": song_up,
        "lyrics_in": lyrics_in,
        "timed_caption_file": timed_caption_file,
        "lyric_timestamp_status": lyric_timestamp_status,
        "external_export_status": external_export_status,
        "external_export_file": external_export_file,
    }
