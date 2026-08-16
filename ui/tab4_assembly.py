import os
import queue
import shutil
import threading
import time

import gradio as gr
import pandas as pd

import config
from models import sync_video_directory
from video import get_project_renders, get_project_videos
from assembly import assemble_video, assemble_video_with_shot_numbers, assemble_cutting_room_floor
from utils import get_file_path


def build(pm_state, shared_shot_state, current_proj_var, shot_table, song_up, vid_resolution_dropdown, vid_gallery, gallery_paths_state):
    """Build Tab 4: Assembly & Cutting Room. Returns dict of exported components."""

    with gr.Tab("4. Assembly & Cutting Room") as tab4_ui:
        gr.Markdown("### ✂️ Cutting Room & Version Comparison")
        with gr.Row():
            compare_shot_dropdown = gr.Dropdown(label="Select Shot to Compare Versions")
            vid_style_filter_dropdown = gr.Dropdown(choices=["All Styles"], value="All Styles", label="Style Filter")
            with gr.Column():
                with gr.Row():
                    prev_shot_btn = gr.Button("⬅️ Previous Shot")
                    next_shot_btn = gr.Button("➡️ Next Shot")
                with gr.Row():
                    prev_multi_btn = gr.Button("⏮ Prev Multi-Version")
                    next_multi_btn = gr.Button("⏭ Next Multi-Version")

        compare_cols = []
        compare_vids = []
        compare_set_btns = []
        compare_cut_btns = []
        compare_paths = []

        with gr.Row():
            for i in range(5):
                with gr.Column(visible=False) as col:
                    cvid = gr.Video(label=f"Version {i+1}", loop=True, interactive=False)
                    cset = gr.Button("⭐ Set as Active", variant="primary")
                    ccut = gr.Button("✂️ Move to Cutting Room Floor", variant="stop")
                    cpath = gr.State("")

                    compare_cols.append(col)
                    compare_vids.append(cvid)
                    compare_set_btns.append(cset)
                    compare_cut_btns.append(ccut)
                    compare_paths.append(cpath)

        gr.Markdown("---")
        gr.Markdown("### 🎞️ Final Assembly")
        with gr.Row():
            assemble_btn = gr.Button("🔢 Assemble with Shot Numbers", variant="secondary")
            assemble_current_btn = gr.Button("Assemble videos with black fallback", variant="primary")
        assembly_progress = gr.Slider(
            minimum=0, maximum=100, value=0, step=1,
            label="Assembly Progress", interactive=False,
        )
        final_video_out = gr.Video(label="Final Cut")
        assembly_status = gr.Textbox(
            label="Assembly Status", interactive=False, lines=1, max_lines=1,
        )

        gr.Markdown("---")
        gr.Markdown("### 🗂️ Cutting Room Floor Compilation")
        gr.Markdown("Assembles every version of every shot (active + discarded) into a single video, grouped by shot in order, oldest version first.")
        with gr.Row():
            assemble_crf_btn = gr.Button("🗂️ Assemble Cutting Room Floor", variant="secondary")
            crf_audio_dropdown = gr.Dropdown(
                choices=["Attach Full Song (Once)", "Loop Full Song", "Use LTX Clip Audio"],
                value="Attach Full Song (Once)",
                label="Audio Mode",
            )

        gr.Markdown("---")
        gr.Markdown("### Previous Renders")
        renders_gallery = gr.Gallery(label="Rendered Videos", columns=4, height="auto", allow_preview=False)
        renders_state = gr.State([])
        with gr.Row():
            render_select_dropdown = gr.Dropdown(label="Select Render to Play", choices=[], interactive=True)
        render_playback = gr.Video(label="Render Playback", interactive=False)

    # --- Tab 4 Internal Events ---

    def update_comparison_view(shot_id, style_filter, pm):
        if not shot_id or pm.df.empty:
            return [gr.update(visible=False)] * 5 + [gr.update(value=None)] * 5 + [""] * 5

        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx:
            return [gr.update(visible=False)] * 5 + [gr.update(value=None)] * 5 + [""] * 5

        paths_str = pm.df.loc[row_idx[0], "All_Video_Paths"]
        if not paths_str or pd.isna(paths_str):
            all_paths = []
        else:
            all_paths = [p.strip() for p in paths_str.split(",") if p.strip() and os.path.exists(p.strip())]

        if not style_filter or style_filter == "All Styles":
            paths = all_paths
        elif style_filter == "No Style":
            paths = [p for p in all_paths if config.slug_from_filename(os.path.basename(p)) is None]
        else:
            slug = config.style_to_slug(style_filter)
            paths = [p for p in all_paths if config.slug_from_filename(os.path.basename(p)) == slug]

        col_updates = []
        vid_updates = []
        path_updates = []

        active_path = pm.df.loc[row_idx[0], "Video_Path"]
        if pd.isna(active_path): active_path = ""

        for i in range(5):
            if i < len(paths):
                p = paths[i]
                is_active = (p == active_path)
                label = f"Version {i+1} {'(ACTIVE)' if is_active else ''}"
                col_updates.append(gr.update(visible=True))
                vid_updates.append(gr.update(value=p, label=label))
                path_updates.append(p)
            else:
                col_updates.append(gr.update(visible=False))
                vid_updates.append(gr.update(value=None))
                path_updates.append("")

        return col_updates + vid_updates + path_updates

    def manual_sync_and_get_choices(pm, shared_shot, progress=gr.Progress()):
        progress(0, desc="Syncing Video Directory...")
        sync_video_directory(pm)
        progress(0.8, desc="Updating Shot List...")
        if pm.df.empty:
            choices = []
        else:
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist()
        progress(0.9, desc="Loading renders...")
        gallery_data, render_paths = get_project_renders(pm)
        render_choices = [os.path.basename(p) for p in render_paths]
        progress(1.0, desc="Complete!")
        value = shared_shot if shared_shot in choices else None
        style_names = config.get_styles_in_videos_dir(pm)
        style_choices = ["All Styles"] + style_names + (["No Style"] if style_names else [])
        return (gr.update(choices=choices, value=value), pm.df, gallery_data, render_paths,
                gr.update(choices=render_choices, value=None), gr.update(choices=style_choices, value="All Styles"))

    tab4_ui.select(manual_sync_and_get_choices, inputs=[pm_state, shared_shot_state],
                   outputs=[compare_shot_dropdown, shot_table, renders_gallery, renders_state, render_select_dropdown, vid_style_filter_dropdown])

    def get_next_shot(current_shot, pm):
        if pm.df.empty:
            gr.Warning("No project loaded.")
            return gr.update()
        choices = pm.df[pm.df["All_Video_Paths"].fillna("") != ""]["Shot_ID"].dropna().unique().tolist()
        if not choices:
            gr.Warning("No shots with renders found.")
            return gr.update(choices=[], value=None)
        if current_shot not in choices:
            all_shots = pm.df["Shot_ID"].dropna().unique().tolist()
            if current_shot in all_shots:
                gr.Info("Current shot has no renders — navigating to nearest shot with renders.")
                curr_idx = all_shots.index(current_shot)
                for i in range(1, len(all_shots) + 1):
                    check_idx = (curr_idx + i) % len(all_shots)
                    if all_shots[check_idx] in choices:
                        return gr.update(choices=choices, value=all_shots[check_idx])
            else:
                gr.Warning("Current shot no longer exists — jumping to first available.")
            return gr.update(choices=choices, value=choices[0])
        idx = choices.index(current_shot)
        if idx == len(choices) - 1:
            gr.Info("Reached the last rendered shot — wrapping to beginning.")
        next_idx = (idx + 1) % len(choices)
        return gr.update(choices=choices, value=choices[next_idx])

    def get_prev_shot(current_shot, pm):
        if pm.df.empty:
            gr.Warning("No project loaded.")
            return gr.update()
        choices = pm.df[pm.df["All_Video_Paths"].fillna("") != ""]["Shot_ID"].dropna().unique().tolist()
        if not choices:
            gr.Warning("No shots with renders found.")
            return gr.update(choices=[], value=None)
        if current_shot not in choices:
            all_shots = pm.df["Shot_ID"].dropna().unique().tolist()
            if current_shot in all_shots:
                gr.Info("Current shot has no renders — navigating to nearest shot with renders.")
                curr_idx = all_shots.index(current_shot)
                for i in range(1, len(all_shots) + 1):
                    check_idx = (curr_idx - i) % len(all_shots)
                    if all_shots[check_idx] in choices:
                        return gr.update(choices=choices, value=all_shots[check_idx])
            else:
                gr.Warning("Current shot no longer exists — jumping to last available.")
            return gr.update(choices=choices, value=choices[-1])
        idx = choices.index(current_shot)
        if idx == 0:
            gr.Info("Reached the first rendered shot — wrapping to end.")
        prev_idx = (idx - 1) % len(choices)
        return gr.update(choices=choices, value=choices[prev_idx])

    prev_shot_btn.click(get_prev_shot, inputs=[compare_shot_dropdown, pm_state], outputs=[compare_shot_dropdown])
    next_shot_btn.click(get_next_shot, inputs=[compare_shot_dropdown, pm_state], outputs=[compare_shot_dropdown])

    def get_multi_version_shots(pm):
        result = []
        for _, row in pm.df.iterrows():
            paths_str = str(row.get("All_Video_Paths", ""))
            paths = [p for p in paths_str.split(",") if p.strip()]
            if len(paths) > 1:
                result.append(row["Shot_ID"])
        return result

    def get_next_multi_shot(current_shot, pm):
        if pm.df.empty: return gr.update()
        shots = get_multi_version_shots(pm)
        if not shots: return gr.update(value=None)
        if current_shot not in shots:
            return gr.update(value=shots[0])
        idx = shots.index(current_shot)
        return gr.update(value=shots[(idx + 1) % len(shots)])

    def get_prev_multi_shot(current_shot, pm):
        if pm.df.empty: return gr.update()
        shots = get_multi_version_shots(pm)
        if not shots: return gr.update(value=None)
        if current_shot not in shots:
            return gr.update(value=shots[-1])
        idx = shots.index(current_shot)
        return gr.update(value=shots[(idx - 1) % len(shots)])

    prev_multi_btn.click(get_prev_multi_shot, inputs=[compare_shot_dropdown, pm_state], outputs=[compare_shot_dropdown])
    next_multi_btn.click(get_next_multi_shot, inputs=[compare_shot_dropdown, pm_state], outputs=[compare_shot_dropdown])

    compare_shot_dropdown.change(update_comparison_view, inputs=[compare_shot_dropdown, vid_style_filter_dropdown, pm_state], outputs=compare_cols + compare_vids + compare_paths)
    compare_shot_dropdown.change(lambda s: s, inputs=[compare_shot_dropdown], outputs=[shared_shot_state])
    vid_style_filter_dropdown.change(update_comparison_view, inputs=[compare_shot_dropdown, vid_style_filter_dropdown, pm_state], outputs=compare_cols + compare_vids + compare_paths)

    def filter_shots_by_style(style_name, pm):
        if style_name == "All Styles" or pm.df.empty:
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist() if not pm.df.empty else []
        else:
            slug = config.style_to_slug(style_name) if style_name != "No Style" else None
            choices = []
            for _, row in pm.df.iterrows():
                paths = str(row.get("All_Video_Paths", "")).split(",")
                for p in paths:
                    file_slug = config.slug_from_filename(os.path.basename(p.strip()))
                    if (slug is None and file_slug is None) or file_slug == slug:
                        choices.append(row["Shot_ID"])
                        break
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value)

    vid_style_filter_dropdown.change(filter_shots_by_style, inputs=[vid_style_filter_dropdown, pm_state], outputs=[compare_shot_dropdown])

    def set_active_video(path, shot_id, style_filter, pm):
        if not path or not os.path.exists(path): return update_comparison_view(shot_id, style_filter, pm)
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if row_idx:
            pm.df.at[row_idx[0], "Video_Path"] = path
            pm.save_data()
        return update_comparison_view(shot_id, style_filter, pm)

    def move_to_cutting_room(path, shot_id, style_filter, proj, pm):
        if not path or not os.path.exists(path):
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist() if not pm.df.empty else []
            return [gr.update(choices=choices, value=shot_id)] + update_comparison_view(shot_id, style_filter, pm) + [gr.update(), gr.update()]

        cut_dir = pm.get_path("cutting_room")
        os.makedirs(cut_dir, exist_ok=True)
        fname = os.path.basename(path)
        dest = os.path.join(cut_dir, fname)
        moved = False
        for attempt in range(5):
            try:
                shutil.move(path, dest)
                moved = True
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.3)
        if not moved:
            # File still locked after retries (e.g. ffprobe hold) — return current view unchanged
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist() if not pm.df.empty else []
            return [gr.update(choices=choices, value=shot_id)] + update_comparison_view(shot_id, style_filter, pm) + [gr.update(), gr.update()]
        sync_video_directory(pm)

        choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist() if not pm.df.empty else []
        if shot_id not in choices:
            shot_id = choices[0] if choices else None

        gal = get_project_videos(pm, proj)
        paths = [item[0] for item in gal]
        return [gr.update(choices=choices, value=shot_id)] + update_comparison_view(shot_id, style_filter, pm) + [gal, paths]

    for i in range(5):
        compare_set_btns[i].click(set_active_video, inputs=[compare_paths[i], compare_shot_dropdown, vid_style_filter_dropdown, pm_state], outputs=compare_cols + compare_vids + compare_paths)
        compare_cut_btns[i].click(move_to_cutting_room, inputs=[compare_paths[i], compare_shot_dropdown, vid_style_filter_dropdown, current_proj_var, pm_state], outputs=[compare_shot_dropdown] + compare_cols + compare_vids + compare_paths + [vid_gallery, gallery_paths_state])

    def refresh_renders(pm):
        gallery_data, render_paths = get_project_renders(pm)
        choices = [os.path.basename(p) for p in render_paths]
        return gallery_data, render_paths, gr.update(choices=choices, value=None), None

    def play_selected_render(selected_name, render_paths):
        if not selected_name or not render_paths:
            return None
        for p in render_paths:
            if os.path.basename(p) == selected_name:
                return p
        return None

    render_select_dropdown.change(play_selected_render, inputs=[render_select_dropdown, renders_state], outputs=[render_playback])

    def on_render_gallery_select(evt: gr.SelectData, render_paths):
        if evt.index is not None and evt.index < len(render_paths):
            path = render_paths[evt.index]
            return path, os.path.basename(path)
        return None, gr.update()

    renders_gallery.select(on_render_gallery_select, inputs=[renders_state], outputs=[render_playback, render_select_dropdown])

    _assembly_outputs = [
        final_video_out, assembly_status, renders_gallery, renders_state,
        render_select_dropdown, assembly_progress,
    ]

    def _run_assembly_task(task, pm, select_result=False):
        if getattr(pm, "assembly_running", False):
            yield (
                gr.update(), "⚠️ An assembly is already running.", gr.update(), gr.update(),
                gr.update(), gr.update(),
            )
            return

        events = queue.Queue(maxsize=8)
        state = {"result": None, "error": None}
        started = time.monotonic()
        last_activity = started
        latest_fraction = 0.0
        latest_message = "Starting assembly worker..."

        def report(fraction, message):
            event = (float(fraction), str(message), time.monotonic())
            try:
                events.put_nowait(event)
            except queue.Full:
                try:
                    events.get_nowait()
                except queue.Empty:
                    pass
                events.put_nowait(event)

        def worker():
            try:
                state["result"] = task(report)
            except Exception as exc:
                state["error"] = exc
            finally:
                pm.assembly_running = False

        pm.assembly_running = True
        thread = threading.Thread(target=worker, name="synesthesia-assembly", daemon=True)
        thread.start()
        while thread.is_alive() or not events.empty():
            try:
                latest_fraction, latest_message, last_activity = events.get(timeout=0.75)
                while True:
                    latest_fraction, latest_message, last_activity = events.get_nowait()
            except queue.Empty:
                pass

            now = time.monotonic()
            elapsed = now - started
            idle = now - last_activity
            heartbeat = ""
            if idle >= 10:
                heartbeat = (
                    f" | ⚠️ No new progress update for {idle:.0f}s; the assembly worker is still running."
                )
            status = (
                f"{latest_message} | Progress: {latest_fraction * 100:.1f}% | Elapsed: {elapsed:.0f}s | "
                f"Last activity: {idle:.1f}s ago{heartbeat}"
            )
            yield (
                gr.update(), status, gr.update(), gr.update(), gr.update(),
                round(latest_fraction * 100),
            )

        elapsed = time.monotonic() - started
        gallery_data, render_paths = get_project_renders(pm)
        render_choices = [os.path.basename(path) for path in render_paths]
        result = state["result"]
        if state["error"] is not None:
            status = f"❌ Assembly failed after {elapsed:.0f}s: {state['error']}"
            result_path = None
            progress_value = round(latest_fraction * 100)
            selected = None
        elif result and os.path.exists(str(result)):
            status = f"✅ Assembly complete in {elapsed:.0f}s: {os.path.basename(str(result))}"
            result_path = result
            progress_value = 100
            selected = os.path.basename(str(result)) if select_result else None
        else:
            status = f"❌ Assembly stopped after {elapsed:.0f}s: {result}"
            result_path = None
            progress_value = round(latest_fraction * 100)
            selected = None
        yield (
            result_path, status, gallery_data, render_paths,
            gr.update(choices=render_choices, value=selected), progress_value,
        )

    def assemble_and_refresh(song_file, resolution, style_filter, pm):
        song_path = get_file_path(song_file)
        task = lambda report: assemble_video(
            song_path, resolution, pm, fallback_mode=True, style_filter=style_filter,
            progress_callback=report,
        )
        yield from _run_assembly_task(task, pm)

    def assemble_numbered_and_refresh(song_file, resolution, style_filter, pm):
        song_path = get_file_path(song_file)
        task = lambda report: assemble_video_with_shot_numbers(
            song_path, resolution, pm, style_filter=style_filter,
            progress_callback=report,
        )
        yield from _run_assembly_task(task, pm)

    assemble_btn.click(
        assemble_numbered_and_refresh,
        inputs=[song_up, vid_resolution_dropdown, vid_style_filter_dropdown, pm_state],
        outputs=_assembly_outputs, show_progress="hidden",
    )
    assemble_current_btn.click(
        assemble_and_refresh,
        inputs=[song_up, vid_resolution_dropdown, vid_style_filter_dropdown, pm_state],
        outputs=_assembly_outputs, show_progress="hidden",
    )

    def assemble_crf_and_refresh(song_file, resolution, audio_mode, pm):
        song_path = get_file_path(song_file)
        task = lambda report: assemble_cutting_room_floor(
            song_path, resolution, pm, audio_mode=audio_mode, progress_callback=report,
        )
        yield from _run_assembly_task(task, pm, select_result=True)

    assemble_crf_btn.click(
        assemble_crf_and_refresh,
        inputs=[song_up, vid_resolution_dropdown, crf_audio_dropdown, pm_state],
        outputs=_assembly_outputs, show_progress="hidden",
    )

    return {
        "tab4_ui": tab4_ui,
    }
