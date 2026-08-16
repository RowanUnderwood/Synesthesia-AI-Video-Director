"""Safe, unelevated NVIDIA GPU power-limit control for Synesthesia.

Changing a GeForce power limit requires Administrator rights on Windows. The
application writes a nonce-protected request and triggers the
``SynesthesiaAIVideoDirectorGpuPower`` SYSTEM scheduled task installed by
``tools/register_gpu_power_task.ps1``. The elevated helper re-queries and
clamps every value before invoking nvidia-smi.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE / "tools"
HELPER_PS1 = TOOLS_DIR / "gpu_power.ps1"
REGISTER_PS1 = TOOLS_DIR / "register_gpu_power_task.ps1"
TASK_NAME = "SynesthesiaAIVideoDirectorGpuPower"

_PROGRAM_DATA = Path(os.environ.get("ProgramData", str(HERE)))
EXCHANGE_DIR = _PROGRAM_DATA / "Synesthesia AI Video Director" / "GpuPowerExchange"
PLAN_FILE = EXCHANGE_DIR / "gpu_power_plan.json"
RESULT_FILE = EXCHANGE_DIR / "gpu_power_result.json"

GPU_ROLES: Dict[str, str] = {"5090": "5090", "4090": "4090", "3090": "3090"}
DEFAULT_WATTS: Dict[str, int] = {"5090": 450, "4090": 350, "3090": 280}
_QUERY_FIELDS = "index,name,power.limit,power.default_limit,power.min_limit,power.max_limit"
_RESULT_TIMEOUT = 25.0
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _run(args: List[str], timeout: int = 15) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            creationflags=_NO_WINDOW, shell=False,
        )
        return completed.returncode, ((completed.stdout or "") + (completed.stderr or "")).strip()
    except FileNotFoundError:
        return 127, f"{args[0]} not found."
    except Exception as exc:  # pragma: no cover - platform/subprocess dependent
        return 1, str(exc)


def query_gpus() -> List[dict]:
    """Return NVIDIA cards and their current/default/legal power limits."""
    rc, output = _run(
        ["nvidia-smi", f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader,nounits"]
    )
    if rc != 0 or not output:
        return []
    result: List[dict] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            result.append({
                "index": int(parts[0]), "name": parts[1], "limit": float(parts[2]),
                "default_limit": float(parts[3]), "min_limit": float(parts[4]),
                "max_limit": float(parts[5]),
            })
        except (TypeError, ValueError):
            continue
    return result


def resolve_roles(gpus: Optional[List[dict]] = None) -> Dict[str, dict]:
    """Resolve logical card roles by model name, never by CUDA/NVML index."""
    cards = query_gpus() if gpus is None else gpus
    found: Dict[str, dict] = {}
    used_indices = set()
    for role, needle in GPU_ROLES.items():
        for card in cards:
            if card["index"] not in used_indices and needle in card["name"]:
                found[role] = card
                used_indices.add(card["index"])
                break
    return found


def role_bounds(role: str, gpus: Optional[List[dict]] = None) -> Tuple[int, int, int]:
    card = resolve_roles(gpus).get(role)
    if not card:
        return 100, 600, DEFAULT_WATTS.get(role, 350)
    return (int(math.ceil(card["min_limit"])), int(math.floor(card["max_limit"])),
            int(round(card["default_limit"])))


def clamp_watts(role: str, value, gpus: Optional[List[dict]] = None) -> int:
    low, high, _stock = role_bounds(role, gpus)
    try:
        watts = int(round(float(value)))
    except (TypeError, ValueError):
        watts = DEFAULT_WATTS.get(role, low)
    return max(low, min(high, watts))


def is_capped(gpus: Optional[List[dict]] = None) -> bool:
    cards = query_gpus() if gpus is None else gpus
    return any(card["limit"] < card["default_limit"] - 1.0 for card in cards)


def current_state() -> str:
    cards = query_gpus()
    if not cards:
        return "⚠️ nvidia-smi unavailable — power limits cannot be read."
    roles = resolve_roles(cards)
    if not roles:
        return "⚠️ Supported GPUs not found: " + ", ".join(card["name"] for card in cards)
    parts = [
        f"{role}: {card['limit']:.0f}/{card['default_limit']:.0f} W "
        f"(legal {card['min_limit']:.0f}–{card['max_limit']:.0f} W)"
        for role, card in roles.items()
    ]
    return " · ".join(parts) + (" — capped" if is_capped(cards) else " — stock")


def helper_installed() -> bool:
    if os.name != "nt":
        return False
    rc, output = _run(["schtasks", "/query", "/tn", TASK_NAME])
    return rc == 0 or "access is denied" in output.lower()


def helper_status() -> str:
    return "✅ Helper registered" if helper_installed() else "⚠️ Helper not registered"


def register_helper() -> Tuple[bool, str]:
    if os.name != "nt":
        return False, "GPU power limits are only supported on Windows."
    if not REGISTER_PS1.exists():
        return False, f"Missing {REGISTER_PS1}."
    try:
        import ctypes
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "powershell.exe",
            f'-NoProfile -ExecutionPolicy Bypass -File "{REGISTER_PS1}"', str(HERE), 1,
        )
        if rc <= 32:
            if rc == 5:
                return False, "❌ Administrator prompt was declined."
            return False, f"❌ Could not launch helper registration (code {rc})."
    except Exception as exc:  # pragma: no cover - Windows shell integration
        return False, f"❌ Registration failed: {exc}"
    for _ in range(80):
        if _run(["schtasks", "/query", "/tn", TASK_NAME])[0] == 0:
            return True, "✅ Power-limit helper registered. No further UAC prompts are required."
        time.sleep(0.25)
    return False, "⚠️ Registration did not finish. Run tools\\register_gpu_power_task.bat manually."


def _dispatch(mode: str, targets: List[dict]) -> Tuple[bool, str]:
    if os.name != "nt":
        return False, "GPU power limits are only supported on Windows."
    if not HELPER_PS1.exists():
        return False, f"Missing {HELPER_PS1}."
    if not helper_installed():
        return False, "⚠️ Register the power-limit helper in Tab 5 first."
    nonce = uuid.uuid4().hex
    try:
        EXCHANGE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            RESULT_FILE.unlink()
        except OSError:
            pass
        PLAN_FILE.write_text(
            json.dumps({"nonce": nonce, "mode": mode, "targets": targets}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"❌ Could not write the GPU power request: {exc}"
    rc, output = _run(["schtasks", "/run", "/tn", TASK_NAME])
    if rc != 0:
        return False, f"❌ Could not start the power-limit helper: {output}"
    deadline = time.time() + _RESULT_TIMEOUT
    result = None
    while time.time() < deadline:
        try:
            candidate = json.loads(RESULT_FILE.read_text(encoding="utf-8-sig"))
            if candidate.get("nonce") == nonce:
                result = candidate
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    if result is None:
        return False, f"⚠️ Power-limit helper did not report back within {_RESULT_TIMEOUT:.0f}s."
    if not result.get("ok"):
        error = result.get("error") or "; ".join(
            f"{entry.get('name', '?')}: {entry.get('message', '')}"
            for entry in result.get("entries", []) if not entry.get("ok")
        )
        return False, f"❌ GPU power {mode} failed: {error or 'unknown error'}"
    summary = " · ".join(
        f"{_short(entry.get('name', '?'))} {entry.get('applied')} W"
        for entry in result.get("entries", [])
    )
    return True, f"⚡ GPU power {'capped' if mode == 'apply' else 'restored'}: {summary}"


def _short(name: str) -> str:
    return next((role for role in GPU_ROLES if role in name), name)


def apply_limits(watts_by_role: Dict[str, int]) -> Tuple[bool, str]:
    cards = query_gpus()
    roles = resolve_roles(cards)
    if not roles:
        return False, "⚠️ No supported GPU found — power limits not applied."
    targets = [
        {"index": card["index"], "watts": clamp_watts(role, watts_by_role[role], cards)}
        for role, card in roles.items() if role in watts_by_role
    ]
    if not targets:
        return False, "⚠️ No wattages configured — power limits not applied."
    return _dispatch("apply", targets)


def restore_defaults() -> Tuple[bool, str]:
    return _dispatch("restore", [])


def watts_from_settings(settings: dict) -> Dict[str, int]:
    cards = query_gpus()
    return {
        role: clamp_watts(role, settings.get(f"power_watts_{role}", default), cards)
        for role, default in DEFAULT_WATTS.items()
    }


def reconcile_on_start(settings: dict) -> Optional[str]:
    """Apply saved caps or clear stale caps left by an interrupted prior run."""
    if os.name != "nt":
        return None
    cards = query_gpus()
    if not cards:
        return None
    if settings.get("power_limit_mode", "no_limit") == "wattage_cap":
        if not helper_installed():
            return "[GPU] Wattage cap is enabled, but the helper is not registered."
        _ok, message = apply_limits(watts_from_settings(settings))
        return f"[GPU] {message}"
    if is_capped(cards) and helper_installed():
        _ok, message = restore_defaults()
        return f"[GPU] Cleared stale power caps. {message}"
    return None


if __name__ == "__main__":
    print(current_state())
    print(helper_status())
    for role, card in resolve_roles().items():
        print(
            f"{role}: nvidia-smi index {card['index']} ({card['name']}), "
            f"{card['min_limit']:.0f}–{card['max_limit']:.0f} W, "
            f"stock {card['default_limit']:.0f} W"
        )
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "apply":
        print(apply_limits(DEFAULT_WATTS)[1])
    elif command == "restore":
        print(restore_defaults()[1])
