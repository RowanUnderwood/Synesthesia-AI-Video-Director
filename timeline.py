import os
import math
import random

import pandas as pd
from pydub import AudioSegment, silence

import config
from utils import get_ltx_duration, snap_to_frame

# ==========================================
# LOGIC: TIMELINE & CONCEPTS
# ==========================================

def get_existing_projects():
    base_dir = "projects"
    if not os.path.exists(base_dir): return []
    projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(projects)

def _backend_max_duration():
    return 15 if config.VIDEO_BACKEND == "MiniMax H3" else 10

def _timeline_duration(seconds, fps=24.0):
    """Keep H3 project timing on the exact 24-FPS timeline grid.

    Its 17n+5 render expansion is applied only immediately before generation.
    """
    return snap_to_frame(seconds, fps) if config.VIDEO_BACKEND == "MiniMax H3" else get_ltx_duration(seconds, fps)

def scan_vocals_advanced(vocals_file_path, project_name, min_silence, silence_thresh, shot_mode, min_dur, max_dur, pm):
    if not project_name or not vocals_file_path or not os.path.exists(vocals_file_path): return pd.DataFrame()

    try:
        audio = AudioSegment.from_file(vocals_file_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return pd.DataFrame()

    total_duration = audio.duration_seconds
    nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=int(min_silence), silence_thresh=silence_thresh)

    new_rows = []
    current_cursor = 0.0
    shot_counter = 1
    fps = 24.0

    MIN_LTX_DUR = _timeline_duration(1.0, fps)

    def create_row(sType, start, end, current_count):
        dur = end - start
        start_frame = round(start * fps)
        end_frame = round(end * fps)
        total_frames = end_frame - start_frame

        return {
            "Shot_ID": f"S{current_count:03d}",
            "Type": sType,
            "Start_Time": float(f"{start:.4f}"),
            "End_Time": float(f"{end:.4f}"),
            "Duration": float(f"{dur:.4f}"),
            "Start_Frame": int(start_frame),
            "End_Frame": int(end_frame),
            "Total_Frames": int(total_frames),
            "Status": "Pending"
        }

    for (start_ms, end_ms) in nonsilent_ranges:
        voc_start = start_ms / 1000.0
        voc_end = end_ms / 1000.0

        gap = voc_start - current_cursor

        while gap >= MIN_LTX_DUR:
            max_safe_int = int(math.floor(gap))
            if max_safe_int < 1:
                break

            chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
            chosen_int = int(math.ceil(chosen_raw))

            if chosen_int > max_safe_int: chosen_int = max_safe_int
            if chosen_int > _backend_max_duration(): chosen_int = _backend_max_duration()

            actual_dur = _timeline_duration(chosen_int, fps)

            if actual_dur > gap:
                break

            new_rows.append(create_row("Action", current_cursor, current_cursor + actual_dur, shot_counter))
            shot_counter += 1
            current_cursor += actual_dur
            gap = voc_start - current_cursor

        vocal_req_dur = voc_end - current_cursor

        while vocal_req_dur > 0:
            if vocal_req_dur > max_dur:
                chosen_int = int(math.ceil(max_dur))
            else:
                chosen_int = int(math.ceil(vocal_req_dur))
                if chosen_int < 1: chosen_int = 1
            if chosen_int > _backend_max_duration(): chosen_int = _backend_max_duration()

            actual_dur = _timeline_duration(chosen_int, fps)

            new_rows.append(create_row("Vocal", current_cursor, current_cursor + actual_dur, shot_counter))
            shot_counter += 1
            current_cursor += actual_dur
            vocal_req_dur = voc_end - current_cursor

    remaining_time = total_duration - current_cursor
    while remaining_time >= MIN_LTX_DUR:
        max_safe_int = int(math.floor(remaining_time))
        if max_safe_int < 1: break

        chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
        chosen_int = int(math.ceil(chosen_raw))

        if chosen_int > max_safe_int: chosen_int = max_safe_int
        if chosen_int > _backend_max_duration(): chosen_int = _backend_max_duration()

        actual_dur = _timeline_duration(chosen_int, fps)
        if actual_dur > remaining_time: break

        new_rows.append(create_row("Action", current_cursor, current_cursor + actual_dur, shot_counter))
        shot_counter += 1
        current_cursor += actual_dur
        remaining_time = total_duration - current_cursor

    if remaining_time > 0.1:
        chosen_int = max(1, min(int(math.ceil(remaining_time)), _backend_max_duration()))
        actual_dur = (snap_to_frame(remaining_time, fps)
                      if config.VIDEO_BACKEND == "MiniMax H3"
                      else _timeline_duration(chosen_int, fps))
        new_rows.append(create_row("Action", current_cursor, current_cursor + actual_dur, shot_counter))

    new_df = pd.DataFrame(new_rows)
    for col in config.REQUIRED_COLUMNS:
        if col not in new_df.columns: new_df[col] = ""

    pm.df = new_df
    pm.save_data()
    return pm.df

def build_simple_timeline(total_duration, shot_type, shot_mode, min_dur, max_dur, pm):
    """Build a timeline of uniform shot type (all Vocal or all Action) without silence scanning."""
    new_rows = []
    current_cursor = 0.0
    shot_counter = 1
    fps = 24.0
    MIN_LTX_DUR = _timeline_duration(1.0, fps)

    def create_row(sType, start, end, current_count):
        dur = end - start
        start_frame = round(start * fps)
        end_frame = round(end * fps)
        total_frames = end_frame - start_frame
        return {
            "Shot_ID": f"S{current_count:03d}",
            "Type": sType,
            "Start_Time": float(f"{start:.4f}"),
            "End_Time": float(f"{end:.4f}"),
            "Duration": float(f"{dur:.4f}"),
            "Start_Frame": int(start_frame),
            "End_Frame": int(end_frame),
            "Total_Frames": int(total_frames),
            "Status": "Pending"
        }

    remaining_time = total_duration - current_cursor
    while remaining_time >= MIN_LTX_DUR:
        max_safe_int = int(math.floor(remaining_time))
        if max_safe_int < 1:
            break

        chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
        chosen_int = int(math.ceil(chosen_raw))

        if chosen_int > max_safe_int:
            chosen_int = max_safe_int
        if chosen_int > _backend_max_duration():
            chosen_int = _backend_max_duration()

        actual_dur = _timeline_duration(chosen_int, fps)
        if actual_dur > remaining_time:
            break

        new_rows.append(create_row(shot_type, current_cursor, current_cursor + actual_dur, shot_counter))
        shot_counter += 1
        current_cursor += actual_dur
        remaining_time = total_duration - current_cursor

    # Handle remaining time
    if remaining_time > 0.1:
        chosen_int = max(1, min(int(math.ceil(remaining_time)), _backend_max_duration()))
        actual_dur = (snap_to_frame(remaining_time, fps)
                      if config.VIDEO_BACKEND == "MiniMax H3"
                      else _timeline_duration(chosen_int, fps))
        new_rows.append(create_row(shot_type, current_cursor, current_cursor + actual_dur, shot_counter))

    new_df = pd.DataFrame(new_rows)
    for col in config.REQUIRED_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""

    pm.df = new_df
    pm.save_data()
    return pm.df
