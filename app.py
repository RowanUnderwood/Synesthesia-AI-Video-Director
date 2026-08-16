import os
import sys
import atexit

# Compatibility shim: Pillow 10.0+ removed PIL.Image.ANTIALIAS (replaced by LANCZOS),
# but moviepy 1.x still references it internally during clip.resize().
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# ==========================================
# WINDOWS ASYNCIO PATCH (Fixes WinError 10054)
# ==========================================
if sys.platform.lower() == "win32" or os.name.lower() == "nt":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        def silence_event_loop_closed(func):
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except (RuntimeError, ConnectionResetError):
                    pass
            return wrapper
        _ProactorBasePipeTransport._call_connection_lost = silence_event_loop_closed(_ProactorBasePipeTransport._call_connection_lost)
    except ImportError:
        pass

import keyboard
import config
import gpu_power
from ui import build_app
from utils import restart_application

_gpu_power_restored = False


def _restore_gpu_power_on_exit():
    global _gpu_power_restored
    if _gpu_power_restored:
        return
    _gpu_power_restored = True
    try:
        if gpu_power.is_capped() and gpu_power.helper_installed():
            _ok, message = gpu_power.restore_defaults()
            print(f"[GPU shutdown] {message}")
    except Exception as exc:
        print(f"[GPU shutdown] Could not restore stock limits: {exc}")


if __name__ == "__main__":
    startup_power_message = gpu_power.reconcile_on_start(config.get_machine_settings())
    if startup_power_message:
        print(startup_power_message)
    atexit.register(_restore_gpu_power_on_exit)
    app = build_app()
    try:
        keyboard.add_hotkey('ctrl+r', restart_application)
        print("⌨️  Hotkey Ctrl+R registered for restarting the application. (Ensure your terminal has focus to use)")
    except Exception as e:
        print(f"⚠️ Could not register hotkey 'ctrl+r'. Run script as admin or ensure 'keyboard' module is installed. Error: {e}")

    app.queue()
    try:
        app.launch(allowed_paths=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")])
    finally:
        _restore_gpu_power_on_exit()
