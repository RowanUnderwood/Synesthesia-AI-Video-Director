import os
import glob
import hashlib
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from proglog import ProgressBarLogger
import imageio_ffmpeg

import config
from utils import format_time

_CPU_THREADS = os.cpu_count() or 1


def _report_progress(callback, fraction, message):
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, float(fraction))), str(message))
    except Exception:
        # Progress reporting must never be able to abort an encode.
        pass


class AssemblyProgressLogger(ProgressBarLogger):
    """Translate MoviePy/FFmpeg audio and frame bars into assembly progress."""

    def __init__(self, callback):
        super().__init__(logged_bars=False, min_time_interval=0.25)
        self.progress_callback = callback
        self.started_at = time.time()

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr != "index":
            return
        state = self.bars.get(bar, {})
        total = state.get("total")
        if not total:
            return
        current = max(0, min(int(value), int(total)))
        ratio = current / max(1, int(total))
        elapsed = time.time() - self.started_at
        if bar == "chunk":
            fraction = 0.38 + 0.07 * ratio
            message = f"Encoding audio: chunk {current}/{int(total)} ({ratio:.0%}) — {elapsed:.0f}s"
        elif bar == "t":
            fraction = 0.45 + 0.54 * ratio
            message = f"Encoding video: frame {current}/{int(total)} ({ratio:.0%}) — {elapsed:.0f}s"
        else:
            return
        _report_progress(self.progress_callback, fraction, message)


def prepare_assembly_clip(path, pm, progress_callback=None, fraction=0.03,
                          label="clip", timeout=60):
    """Return a MoviePy-safe path, caching H.264 MP4 proxies for legacy WebMs.

    MiniMax's enhanced video-combine node previously selected AV1/WebM in Auto
    mode. Some of those files require NVIDIA's AV1 decoder and make MoviePy's
    default software reader wait forever for a first frame. Originals are never
    changed; a project-local proxy is atomically created and reused.
    """
    source = os.path.abspath(str(path))
    if os.path.splitext(source)[1].lower() != ".webm":
        return source
    if not os.path.isfile(source):
        raise FileNotFoundError(source)

    cache_dir = pm.get_path("assembly_cache")
    os.makedirs(cache_dir, exist_ok=True)
    source_id = hashlib.sha256(os.path.normcase(source).encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(source))[0]
    target = os.path.join(cache_dir, f"{stem}_{source_id}_h264.mp4")
    if (os.path.isfile(target) and os.path.getsize(target) > 1024
            and os.path.getmtime(target) >= os.path.getmtime(source)):
        _report_progress(progress_callback, fraction, f"Using cached H.264 proxy for {label}...")
        return target

    _report_progress(progress_callback, fraction, f"Converting legacy WebM for {label} to H.264 MP4...")
    temp_target = target + ".part.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    attempts = [
        ("av1_cuvid", "h264_nvenc", ["-preset", "p4", "-cq", "18"]),
        (None, "h264_nvenc", ["-preset", "p4", "-cq", "18"]),
        (None, "libx264", ["-preset", "veryfast", "-crf", "18"]),
    ]
    errors = []
    for decoder, encoder, encoder_options in attempts:
        try:
            if os.path.exists(temp_target):
                os.remove(temp_target)
            command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
            if decoder:
                command.extend(["-c:v", decoder])
            command.extend([
                "-i", source,
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", encoder, *encoder_options,
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                temp_target,
            ])
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode == 0 and os.path.isfile(temp_target) and os.path.getsize(temp_target) > 1024:
                os.replace(temp_target, target)
                _report_progress(progress_callback, fraction, f"H.264 proxy ready for {label}.")
                return target
            detail = (completed.stderr or completed.stdout or "unknown FFmpeg error").strip()
            errors.append(f"{decoder or 'auto'}/{encoder}: {detail[-500:]}")
        except subprocess.TimeoutExpired:
            errors.append(f"{decoder or 'auto'}/{encoder}: timed out after {timeout}s")
        except OSError as exc:
            errors.append(f"{decoder or 'auto'}/{encoder}: {exc}")
        _report_progress(
            progress_callback, fraction,
            f"Proxy attempt failed for {label}; trying the next decoder/encoder...",
        )

    if os.path.exists(temp_target):
        try:
            os.remove(temp_target)
        except OSError:
            pass
    message = (
        f"Could not create an assembly proxy for {os.path.basename(source)}. "
        + " | ".join(errors)
    )
    _report_progress(progress_callback, fraction, f"Proxy conversion failed for {label}: {message}")
    raise RuntimeError(message)

def _project_slug(pm):
    """Return a filename-safe lowercase slug from the project name."""
    import re
    name = pm.current_project or ""
    return re.sub(r'[^a-z0-9_]+', '_', name.lower()).strip('_')

def _get_shot_resolution(pm, shot_id, fallback="1080p"):
    """Look up Render_Resolution for a shot from pm.df; fall back if missing/NaN."""
    try:
        row = pm.df[pm.df['Shot_ID'] == shot_id]
        if not row.empty and 'Render_Resolution' in row.columns:
            res = row.iloc[0].get('Render_Resolution')
            if res and str(res).strip() not in ('', 'nan'):
                return str(res)
    except Exception:
        pass
    return fallback


def _render_cost_str(pm, fallback_resolution="1080p", generation_mode="LTX-Native"):
    """Return a filename-safe cost string like '_cost0.08' using per-shot Render_Resolution.
    Falls back to fallback_resolution for shots without a recorded resolution.
    Uses SYSTEM_WATTAGE and ELECTRICITY_COST from config. Returns '' on any error."""
    try:
        df = pm.df
        if df.empty:
            return ""
        has_video = df['Video_Path'].notna() & (df['Video_Path'].astype(str).str.strip() != "")
        rendered = df[has_video].copy()
        if rendered.empty:
            return ""
        total_render_time = 0.0
        for _, row in rendered.iterrows():
            res_val = row.get('Render_Resolution') if 'Render_Resolution' in row.index and pd.notna(row.get('Render_Resolution')) else None
            res = str(res_val).strip() if res_val and str(res_val).strip() not in ('', 'nan') else fallback_resolution
            dur = float(row['Duration'])
            total_render_time += config.estimate_render_seconds(dur, res, generation_mode)
        cost = (total_render_time / 3600.0) * (config.SYSTEM_WATTAGE / 1000.0) * config.ELECTRICITY_COST
        return f"_cost{cost:.2f}"
    except Exception:
        return ""

# ==========================================
# LOGIC: VIDEO ASSEMBLY
# ==========================================

def assemble_video(full_song_path, resolution, pm, fallback_mode=False, style_filter=None,
                   progress_callback=None):
    df = pm.df
    clips = []
    clips_to_close = []
    if df.empty: return "No shots to assemble."
    _report_progress(progress_callback, 0.01, "Reading timeline and locating active clips...")

    df = df.sort_values(by="Start_Time")
    expected_cursor = 0.0

    # Resolve the style slug to use for filtering (None = no filter = use Video_Path)
    filter_slug = None
    filter_no_style = False
    if style_filter and style_filter not in (None, "All Styles"):
        if style_filter == "No Style":
            filter_no_style = True
        else:
            filter_slug = config.style_to_slug(style_filter)

    def pick_vid_path(row):
        if filter_slug is not None or filter_no_style:
            all_paths = [p.strip() for p in str(row.get("All_Video_Paths", "")).split(",") if p.strip()]
            if filter_no_style:
                matching = [p for p in all_paths if config.slug_from_filename(os.path.basename(p)) is None]
            else:
                matching = [p for p in all_paths if config.slug_from_filename(os.path.basename(p)) == filter_slug]
            return matching[0] if matching else None
        return row.get('Video_Path')

    # Detect target resolution from the first available video clip
    # LTX output resolution varies (multiples of 32, differs with/without audio)
    target_size = None
    for _, r in df.iterrows():
        vp = pick_vid_path(r)
        if vp and pd.notna(vp) and os.path.exists(str(vp)):
            probe = None
            try:
                probe_path = prepare_assembly_clip(
                    vp, pm, progress_callback, 0.03,
                    f"resolution probe {r.get('Shot_ID', '?')}",
                )
                probe = VideoFileClip(probe_path)
                target_size = tuple(probe.size)
                break
            except Exception as exc:
                print(f"Assembly resolution probe skipped {vp}: {exc}")
            finally:
                if probe is not None:
                    probe.close()
    if target_size is None:
        target_size = config.RESOLUTION_MAP.get(resolution, (1920, 1080))

    total_shots = len(df)
    for seq_num, (index, row) in enumerate(df.iterrows(), start=1):
        _report_progress(
            progress_callback, 0.05 + 0.30 * (seq_num / max(1, total_shots)),
            f"Preparing clip {seq_num}/{total_shots}: {row.get('Shot_ID', '?')}",
        )
        vid_path = pick_vid_path(row)
        dur = float(row['Duration'])
        start_time = float(row['Start_Time'])
        # The timeline's frame count is authoritative. H3 grid-padding frames
        # are deliberately rendered but never allowed to lengthen assembly.
        try:
            snapped_dur = int(row.get('Total_Frames')) / 24
        except (TypeError, ValueError):
            snapped_dur = round(dur * 24) / 24
        clip = None

        gap = round((start_time - expected_cursor) * 24) / 24
        if gap > 0.05:
            pad = ColorClip(size=target_size, color=(0,0,0), duration=gap).set_fps(24)
            clips.append(pad)
            clips_to_close.append(pad)

        if vid_path and pd.notna(vid_path) and os.path.exists(str(vid_path)):
            try:
                prepared_path = prepare_assembly_clip(
                    vid_path, pm, progress_callback,
                    0.05 + 0.30 * (seq_num / max(1, total_shots)),
                    str(row.get('Shot_ID', '?')),
                )
                clip = VideoFileClip(prepared_path).without_audio().set_fps(24)

                if clip.duration > snapped_dur:
                    clip = clip.subclip(0, snapped_dur)
                clip = clip.set_duration(snapped_dur)

                if tuple(clip.size) != tuple(target_size):
                    clip = clip.resize(newsize=target_size)

            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")

        if clip is None:
            if fallback_mode:
                clip = ColorClip(size=target_size, color=(0,0,0), duration=snapped_dur).set_fps(24)
            else:
                for c in clips_to_close: c.close()
                return f"Error: Missing or corrupt video for shot at {start_time}s. Assembly stopped (Strict Mode)."

        if clip is not None:
            clips.append(clip)
            clips_to_close.append(clip)

        expected_cursor = start_time + snapped_dur

    if not clips: return "No valid clips found."

    final = concatenate_videoclips(clips, method="chain")
    audio = None

    audio_path = full_song_path if (full_song_path and os.path.exists(full_song_path)) else pm.get_asset_path_if_exists("full_song.mp3")
    if not audio_path: audio_path = pm.get_asset_path_if_exists("vocals.mp3")

    if audio_path and os.path.exists(audio_path):
        try:
            _report_progress(progress_callback, 0.37, "Attaching the project audio track...")
            audio = AudioFileClip(audio_path)
            if audio.duration > final.duration: audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)
        except Exception as e: print(f"Audio attach failed: {e}")

    total_seconds = pm.get_current_total_time()
    time_str = format_time(total_seconds)

    style_part = ""
    if filter_slug:
        style_part = f"_{filter_slug}"
    elif filter_no_style:
        style_part = "_no_style"
    slug = _project_slug(pm)
    cost_part = _render_cost_str(pm, fallback_resolution=resolution)
    out_path = os.path.join(pm.get_path("renders"), f"{slug}_final_cut{style_part}{cost_part}_{time_str}.mp4")

    try:
        _report_progress(progress_callback, 0.42, "Starting FFmpeg video encode...")
        final.write_videofile(
            out_path, fps=24, codec='libx264', audio_codec='aac',
            temp_audiofile=os.path.join(pm.get_path("renders"), "temp_audio.m4a"),
            remove_temp=True, threads=_CPU_THREADS,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-ar", "44100"],
            logger=AssemblyProgressLogger(progress_callback),
        )
    finally:
        final.close()
        if audio is not None:
            try: audio.close()
            except Exception: pass
        for c in clips_to_close:
            try: c.close()
            except Exception: pass

    _report_progress(progress_callback, 1.0, "Assembly complete.")
    return out_path

def _make_shot_label_clip(shot_id, seq_num, total_shots, size, duration, fps=24):
    w, h = size
    font_size = max(24, h // 22)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    text = f"{shot_id}  [{seq_num}/{total_shots}]"

    tmp = Image.new("RGBA", (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = 10
    box = [pad, pad, pad + tw + pad * 2, pad + th + pad * 2]
    draw.rectangle(box, fill=(0, 0, 0, 160))
    draw.text((pad * 2, int(pad * 1.5)), text, fill=(255, 220, 0, 255), font=font)

    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0

    label = ImageClip(rgb, ismask=False).set_duration(duration).set_fps(fps)
    mask = ImageClip(alpha, ismask=True).set_duration(duration).set_fps(fps)
    return label.set_mask(mask)


def assemble_video_with_shot_numbers(full_song_path, resolution, pm, style_filter=None,
                                     progress_callback=None):
    df = pm.df
    clips = []
    clips_to_close = []
    if df.empty: return "No shots to assemble."
    _report_progress(progress_callback, 0.01, "Reading timeline and locating clips for numbered review...")

    df = df.sort_values(by="Start_Time")
    expected_cursor = 0.0

    filter_slug = None
    filter_no_style = False
    if style_filter and style_filter not in (None, "All Styles"):
        if style_filter == "No Style":
            filter_no_style = True
        else:
            filter_slug = config.style_to_slug(style_filter)

    def pick_vid_path(row):
        if filter_slug is not None or filter_no_style:
            all_paths = [p.strip() for p in str(row.get("All_Video_Paths", "")).split(",") if p.strip()]
            if filter_no_style:
                matching = [p for p in all_paths if config.slug_from_filename(os.path.basename(p)) is None]
            else:
                matching = [p for p in all_paths if config.slug_from_filename(os.path.basename(p)) == filter_slug]
            return matching[0] if matching else None
        return row.get('Video_Path')

    target_size = None
    for _, r in df.iterrows():
        vp = pick_vid_path(r)
        if vp and pd.notna(vp) and os.path.exists(str(vp)):
            probe = None
            try:
                probe_path = prepare_assembly_clip(
                    vp, pm, progress_callback, 0.03,
                    f"resolution probe {r.get('Shot_ID', '?')}",
                )
                probe = VideoFileClip(probe_path)
                target_size = tuple(probe.size)
                break
            except Exception as exc:
                print(f"Numbered assembly resolution probe skipped {vp}: {exc}")
            finally:
                if probe is not None:
                    probe.close()
    if target_size is None:
        target_size = config.RESOLUTION_MAP.get(resolution, (1920, 1080))

    total_shots = len(df)
    seq_num = 0

    for index, row in df.iterrows():
        seq_num += 1
        _report_progress(
            progress_callback, 0.05 + 0.30 * (seq_num / max(1, total_shots)),
            f"Preparing numbered clip {seq_num}/{total_shots}: {row.get('Shot_ID', '?')}",
        )
        vid_path = pick_vid_path(row)
        dur = float(row['Duration'])
        start_time = float(row['Start_Time'])
        try:
            snapped_dur = int(row.get('Total_Frames')) / 24
        except (TypeError, ValueError):
            snapped_dur = round(dur * 24) / 24
        clip = None

        gap = round((start_time - expected_cursor) * 24) / 24
        if gap > 0.05:
            pad = ColorClip(size=target_size, color=(0,0,0), duration=gap).set_fps(24)
            clips.append(pad)
            clips_to_close.append(pad)

        if vid_path and pd.notna(vid_path) and os.path.exists(str(vid_path)):
            try:
                prepared_path = prepare_assembly_clip(
                    vid_path, pm, progress_callback,
                    0.05 + 0.30 * (seq_num / max(1, total_shots)),
                    str(row.get('Shot_ID', '?')),
                )
                clip = VideoFileClip(prepared_path).without_audio().set_fps(24)
                if clip.duration > snapped_dur:
                    clip = clip.subclip(0, snapped_dur)
                clip = clip.set_duration(snapped_dur)
                if tuple(clip.size) != tuple(target_size):
                    clip = clip.resize(newsize=target_size)
            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")

        if clip is None:
            clip = ColorClip(size=target_size, color=(0,0,0), duration=snapped_dur).set_fps(24)

        label = _make_shot_label_clip(row["Shot_ID"], seq_num, total_shots, target_size, clip.duration)
        clip = CompositeVideoClip([clip, label])

        clips.append(clip)
        clips_to_close.append(clip)
        expected_cursor = start_time + snapped_dur

    if not clips: return "No valid clips found."

    final = concatenate_videoclips(clips, method="chain")
    audio = None

    audio_path = full_song_path if (full_song_path and os.path.exists(full_song_path)) else pm.get_asset_path_if_exists("full_song.mp3")
    if not audio_path: audio_path = pm.get_asset_path_if_exists("vocals.mp3")

    if audio_path and os.path.exists(audio_path):
        try:
            _report_progress(progress_callback, 0.37, "Attaching the project audio track...")
            audio = AudioFileClip(audio_path)
            if audio.duration > final.duration: audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)
        except Exception as e: print(f"Audio attach failed: {e}")

    total_seconds = pm.get_current_total_time()
    time_str = format_time(total_seconds)
    slug = _project_slug(pm)
    cost_part = _render_cost_str(pm, fallback_resolution=resolution)
    out_path = os.path.join(pm.get_path("renders"), f"{slug}_shot_review{cost_part}_{time_str}.mp4")

    try:
        _report_progress(progress_callback, 0.42, "Starting FFmpeg numbered-review encode...")
        final.write_videofile(
            out_path, fps=24, codec='libx264', audio_codec='aac',
            temp_audiofile=os.path.join(pm.get_path("renders"), "temp_audio.m4a"),
            remove_temp=True, threads=_CPU_THREADS,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-ar", "44100"],
            logger=AssemblyProgressLogger(progress_callback),
        )
    finally:
        final.close()
        if audio is not None:
            try: audio.close()
            except Exception: pass
        for c in clips_to_close:
            try: c.close()
            except Exception: pass

    _report_progress(progress_callback, 1.0, "Numbered review assembly complete.")
    return out_path


def assemble_cutting_room_floor(full_song_path, resolution, pm, audio_mode="Attach Full Song (Once)",
                                progress_callback=None):
    """Assemble all versions (cutting_room + active videos) into a single chronological showreel."""
    vid_dir = pm.get_path("videos")
    cut_dir = pm.get_path("cutting_room")

    all_files = []
    for d in [vid_dir, cut_dir]:
        if os.path.exists(d):
            for extension in ("*.mp4", "*.webm", "*.mov", "*.mkv"):
                all_files.extend(glob.glob(os.path.join(d, extension)))

    if not all_files:
        return "No videos found in videos or cutting_room directories."
    _report_progress(progress_callback, 0.01, "Locating active and cutting-room clips...")

    def sort_key(filepath):
        shot_id = os.path.basename(filepath).split("_")[0].upper()
        return (shot_id, os.path.getmtime(filepath))

    all_files.sort(key=sort_key)

    target_size = None
    for f in all_files:
        probe = None
        try:
            probe_path = prepare_assembly_clip(
                f, pm, progress_callback, 0.03,
                f"resolution probe {os.path.basename(f)}",
            )
            probe = VideoFileClip(probe_path)
            target_size = tuple(probe.size)
            break
        except Exception as exc:
            print(f"Cutting-room resolution probe skipped {f}: {exc}")
        finally:
            if probe is not None:
                probe.close()
    if target_size is None:
        target_size = config.RESOLUTION_MAP.get(resolution, (1920, 1080))

    clips = []
    clips_to_close = []
    for seq_num, f in enumerate(all_files, start=1):
        _report_progress(
            progress_callback, 0.05 + 0.30 * (seq_num / max(1, len(all_files))),
            f"Preparing cutting-room clip {seq_num}/{len(all_files)}: {os.path.basename(f)}",
        )
        try:
            prepared_path = prepare_assembly_clip(
                f, pm, progress_callback,
                0.05 + 0.30 * (seq_num / max(1, len(all_files))),
                os.path.basename(f),
            )
            if audio_mode == "Use LTX Clip Audio":
                clip = VideoFileClip(prepared_path).set_fps(24)
            else:
                clip = VideoFileClip(prepared_path).without_audio().set_fps(24)
            if tuple(clip.size) != target_size:
                clip = clip.resize(newsize=target_size)
            clips.append(clip)
            clips_to_close.append(clip)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not clips:
        return "No valid clips could be loaded."

    final = concatenate_videoclips(clips, method="chain")

    audio_path = full_song_path if (full_song_path and os.path.exists(full_song_path)) else pm.get_asset_path_if_exists("full_song.mp3")
    if not audio_path:
        audio_path = pm.get_asset_path_if_exists("vocals.mp3")

    audio = None
    if audio_mode != "Use LTX Clip Audio" and audio_path and os.path.exists(audio_path):
        try:
            _report_progress(progress_callback, 0.37, "Preparing cutting-room audio track...")
            audio = AudioFileClip(audio_path)
            if audio_mode == "Loop Full Song" and audio.duration < final.duration:
                loops = int(final.duration / audio.duration) + 2
                from moviepy.editor import concatenate_audioclips
                audio = concatenate_audioclips([audio] * loops).subclip(0, final.duration)
            elif audio.duration > final.duration:
                audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)
        except Exception as e:
            print(f"Audio attach failed: {e}")

    total_seconds = pm.get_current_total_time()
    time_str = format_time(total_seconds)
    slug = _project_slug(pm)
    crf_cost = 0.0
    for _f in all_files:
        _sid = os.path.basename(_f).split('_')[0]
        _res = _get_shot_resolution(pm, _sid, fallback=resolution)
        try:
            _dur = float(pm.df[pm.df['Shot_ID'] == _sid]['Duration'].values[0])
        except Exception:
            _dur = 3.0
        crf_cost += config.estimate_render_seconds(_dur, _res, "LTX-Native") * (config.SYSTEM_WATTAGE / 1000.0) * config.ELECTRICITY_COST / 3600.0
    cost_part = f"_cost{crf_cost:.2f}" if crf_cost > 0 else ""
    out_path = os.path.join(pm.get_path("renders"), f"{slug}_cutting_room_floor{cost_part}_{time_str}.mp4")

    try:
        _report_progress(progress_callback, 0.42, "Starting FFmpeg cutting-room encode...")
        final.write_videofile(
            out_path, fps=24, codec='libx264', audio_codec='aac',
            temp_audiofile=os.path.join(pm.get_path("renders"), "temp_audio_crf.m4a"),
            remove_temp=True, threads=_CPU_THREADS,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-ar", "44100"],
            logger=AssemblyProgressLogger(progress_callback),
        )
    finally:
        final.close()
        if audio is not None:
            try: audio.close()
            except Exception: pass
        for c in clips_to_close:
            try: c.close()
            except Exception: pass

    _report_progress(progress_callback, 1.0, "Cutting-room assembly complete.")
    return out_path
