import os
import sys
import math
import base64
import subprocess

# ==========================================
# SYSTEM UTILITIES
# ==========================================

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logo_base64 = get_base64_image(os.path.join(_SCRIPT_DIR, "Synesthesiatransparent.png"))

header_html = f'''
<div style="display: flex; align-items: center; gap: 15px;">
    <img src="data:image/png;base64,{logo_base64}" style="height: 80px; min-width: 80px; object-fit: contain;">
    <h1 style="margin: 0; padding-bottom: 5px;">Synesthesia AI Video Director</h1>
</div>
'''

def get_file_path(file_obj):
    """Safely extracts a file path from a Gradio file component."""
    if file_obj is None: return None
    if isinstance(file_obj, str): return file_obj
    if hasattr(file_obj, 'name'): return file_obj.name
    if isinstance(file_obj, dict) and 'name' in file_obj: return file_obj['name']
    return None

def restart_application():
    """Restarts the current python process."""
    print("♻️ Restarting application via hotkey...")
    python = sys.executable
    os.execl(python, python, *sys.argv)

def snap_to_frame(seconds, fps=24):
    frame_dur = 1.0 / fps
    return round(seconds / frame_dur) * frame_dur

def get_ltx_frame_count(target_seconds, fps=24):
    """
    Calculates LTX Desktop-compliant frame counts (locked to whole seconds do not change this).
    1s=25f, 2s=49f, 3s=73f, 4s=97f, 5s=121f, 6s=145f, 7s=169f, 8s=193f, 9s=217f, 10s=241f.
    """
    target_int = int(math.ceil(target_seconds))

    if target_int < 1:
        target_int = 1
    if target_int > 10:         # MAX: 10 seconds (720p or lower only above 5s)
        target_int = 10

    total_frames = target_int * fps
    backend_frames = round((total_frames - 1) / 8) * 8 + 1
    return max(backend_frames, 9)

def get_ltx_duration(seconds, fps=24):
    """
    Returns the true floating-point timeline duration of the locked integer frame counts.
    """
    frames = get_ltx_frame_count(seconds, fps)
    return frames / fps

def get_h3_render_frame_count(timeline_frames):
    """Return H3's 17n+5 render length for a timeline frame count.

    Extra frames are render-only padding.  The project's 24-FPS timeline and
    final assembly remain anchored to the original frame count.
    """
    frames = max(5, int(timeline_frames))
    frames += (5 - frames) % 17
    if frames > 362:
        raise ValueError("MiniMax H3 supports at most 362 frames (15.083 seconds at 24 FPS).")
    return frames

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"

def format_eta(seconds_remaining):
    """Format a remaining-seconds value into a compact human-readable string."""
    if seconds_remaining <= 0:
        return "done"
    m, s = divmod(int(seconds_remaining), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
