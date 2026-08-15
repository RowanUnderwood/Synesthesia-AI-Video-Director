"""
Synesthesia AI Video Director — Launcher
stdlib only: runs with system Python before the venv exists.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.resolve()
os.chdir(PROJECT_DIR)

VENV_PYTHON  = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
VENV_PIP     = PROJECT_DIR / "venv" / "Scripts" / "pip.exe"
CONFIG_FILE  = PROJECT_DIR / "launcher_config.json"

DEV_MODE          = "--dev" in sys.argv
SYNESTHESIA_PORT  = 7860   # default Gradio port; change if you set server_port in app.py

# ── Config template written on first run ─────────────────────────────────────
_CONFIG_TEMPLATE = {
    "lm_studio_path": "C:\\Users\\YourName\\AppData\\Local\\Programs\\LM Studio\\LM Studio.exe",
    "ltx_desktop_path": "H:\\LTX Desktop\\LTX Desktop.exe",
    "ltx_desktop_port": None,
    "stability_matrix_path": "",
    "git_branch": "main",
}

_CONFIG_COMMENTS = """\
// launcher_config.json — Synesthesia launcher user config
// Edit the paths below to match your machine, then re-run the launcher.
//
// lm_studio_path       : full path to LM Studio.exe (or leave blank to skip)
// ltx_desktop_path     : full path to LTX Desktop.exe or LTX Desktop.bat
// ltx_desktop_port     : if set, launcher checks this localhost port instead of
//                        process name to decide whether LTX is already running.
//                        e.g. 4000 for LTX-2 Cinematic Workstation. null = use process name.
// stability_matrix_path: full path to StabilityMatrix.exe (or leave blank to skip)
// git_branch           : "main" = stable release tags  |  "dev" = latest dev commits
//                        (only used by run.bat / launcher without --dev flag)
//
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pause_exit(code: int = 1) -> None:
    if sys.stdin.isatty():
        input("Press Enter to exit...")
    sys.exit(code)


def _run(*args, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run wrapper with consistent defaults."""
    return subprocess.run(args, **kwargs)


def _process_running(exe_name: str) -> bool:
    """Return True if a process with the given image name is currently running."""
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {exe_name}"],
        capture_output=True, text=True,
    )
    return exe_name.lower() in result.stdout.lower()


def _port_in_use(port: int) -> bool:
    """Return True if something is already listening on the given localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        # Write a commented template (JSON doesn't support comments natively,
        # so we prepend them as a block and write valid JSON after).
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(_CONFIG_COMMENTS)
            json.dump(_CONFIG_TEMPLATE, f, indent=4)
            f.write("\n")
        print("=" * 60)
        print("  First-time setup: launcher_config.json has been created.")
        print("=" * 60)
        print()
        print(f"  Edit the file to set your paths, then re-run:")
        print(f"  {CONFIG_FILE}")
        print()
        pause_exit(0)

    # Strip leading comment lines before parsing JSON.
    raw = CONFIG_FILE.read_text(encoding="utf-8")
    json_lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("//")]
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError as exc:
        print(f"ERROR: Could not parse launcher_config.json: {exc}")
        print(f"  Fix the file and try again: {CONFIG_FILE}")
        pause_exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Venv / first-run install
# ─────────────────────────────────────────────────────────────────────────────

def ensure_venv() -> None:
    if VENV_PYTHON.exists():
        return

    print("No installation found. Running first-time setup...")
    print()

    print("Creating virtual environment...")
    result = _run(sys.executable, "-m", "venv", "venv")
    if result.returncode != 0:
        print("ERROR: Failed to create virtual environment.")
        pause_exit(1)

    print("Installing dependencies...")
    result = _run(str(VENV_PIP), "install", "-r", "requirements.txt")
    if result.returncode != 0:
        print("ERROR: Failed to install dependencies.")
        pause_exit(1)

    print()
    print("=" * 60)
    print("  Installation complete!")
    print("=" * 60)
    print()
    print("IMPORTANT: You also need FFmpeg installed and on your PATH.")
    print("Download FFmpeg from https://ffmpeg.org/ if you haven't already.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Git update check  (prod mode only)
# ─────────────────────────────────────────────────────────────────────────────

def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True,
    )


def _git_available() -> bool:
    r = _git("--version")
    return r.returncode == 0


def _pip_upgrade() -> None:
    print("Updating dependencies...")
    result = _run(str(VENV_PIP), "install", "--upgrade", "-r", "requirements.txt")
    if result.returncode != 0:
        print("WARNING: Dependency update failed — some features may not work correctly.")


def check_git(cfg: dict) -> None:
    if not _git_available():
        print("NOTE: Git not found — skipping update check.")
        print()
        return

    branch = cfg.get("git_branch", "main").strip().lower()
    print("Checking for updates...")

    if branch == "dev":
        r = _git("fetch", "origin")
        if r.returncode != 0:
            print("WARNING: Could not reach remote — skipping update check.")
            print()
            return

        local_sha  = _git("rev-parse", "HEAD").stdout.strip()
        remote_sha = _git("rev-parse", "origin/dev").stdout.strip()

        if not remote_sha:
            print("WARNING: Could not determine remote dev state — skipping update check.")
            print()
            return

        if local_sha == remote_sha:
            print("Already up to date.")
            print()
            return

        print("A new dev update is available.")
        answer = input("Update now? (Y/N): ").strip().upper()
        if answer == "Y":
            r = _git("pull", "origin", "dev")
            if r.returncode != 0:
                print("WARNING: Update failed — launching with current version.")
                print()
                return
            _pip_upgrade()
            print()
            print("Update complete.")
            print()
        else:
            print("Skipping update.")
            print()

    else:
        # main — compare release tags
        r = _git("fetch", "--tags", "--force")
        if r.returncode != 0:
            print("WARNING: Could not reach remote — skipping update check.")
            print()
            return

        tags_out = _git("tag", "--sort=-version:refname").stdout.strip().splitlines()
        latest_tag = tags_out[0] if tags_out else ""

        if not latest_tag:
            print("NOTE: No release tags found — skipping update check.")
            print()
            return

        current_tag = _git("describe", "--tags", "--exact-match", "HEAD").stdout.strip()

        if current_tag == latest_tag:
            print(f"Already on latest release ({latest_tag}).")
            print()
            return

        if not current_tag:
            print("Current version: (development / untagged)")
        else:
            print(f"Current version: {current_tag}")
        print(f"Latest release:  {latest_tag}")
        print()

        answer = input(f"Update to {latest_tag}? (Y/N): ").strip().upper()
        if answer == "Y":
            r = _git("checkout", latest_tag)
            if r.returncode != 0:
                print("WARNING: Update failed — launching with current version.")
                print()
                return
            _pip_upgrade()
            print()
            print("Update complete.")
            print()
        else:
            print("Skipping update.")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# External app launchers
# ─────────────────────────────────────────────────────────────────────────────

def launch_if_not_running(exe_name: str, path_str: str, port: int = None) -> None:
    """Launch an external app if it is not already running.

    If `port` is given, presence on that localhost port is used as the
    already-running check instead of exe_name process detection.
    """
    if not path_str:
        return

    # Already-running check.
    if port is not None:
        if _port_in_use(port):
            return
    elif _process_running(exe_name):
        return

    path = Path(path_str)
    if not path.exists():
        print(f"WARNING: Could not launch {path.name} — path not found: {path_str}")
        return

    try:
        if path.suffix.lower() == ".bat":
            # CREATE_NEW_CONSOLE: bat file gets its own visible window.
            # cwd=path.parent: bat file runs in its own directory so relative
            #   paths inside it resolve correctly.
            subprocess.Popen(
                ["cmd", "/c", str(path)],
                cwd=str(path.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            subprocess.Popen(
                [str(path)],
                cwd=str(path.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
    except Exception as exc:
        print(f"WARNING: Failed to launch {path.name}: {exc}")


def _claude_running() -> bool:
    """Return True if Claude (claude.exe or a claude-bearing node process) is running."""
    if _process_running("claude.exe"):
        return True
    # Check for node.exe processes that have "claude" in their command line.
    # Uses PowerShell to avoid the deprecated wmic tool.
    ps_cmd = (
        "Get-WmiObject Win32_Process "
        "| Where-Object { $_.Name -eq 'node.exe' } "
        "| Select-Object -ExpandProperty CommandLine"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, text=True,
    )
    return "claude" in result.stdout.lower()


def launch_claude() -> None:
    """Open a new cmd window running Claude in the project directory (dev mode only)."""
    if _claude_running():
        return
    subprocess.Popen(
        ["cmd", "/k", "claude"],
        cwd=str(PROJECT_DIR),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    mode_tag = "  [DEV MODE - no update check]" if DEV_MODE else ""
    print("=" * 60)
    print(f"  Synesthesia AI Video Director{mode_tag}")
    print("=" * 60)
    print()

    cfg = load_config()
    ensure_venv()

    if DEV_MODE:
        os.startfile(str(PROJECT_DIR))   # open Explorer
        launch_claude()
    else:
        check_git(cfg)

    # Bail out early if Synesthesia is already running.
    if _port_in_use(SYNESTHESIA_PORT):
        print(f"Synesthesia is already running at http://localhost:{SYNESTHESIA_PORT}")
        print("Opening in browser...")
        os.startfile(f"http://localhost:{SYNESTHESIA_PORT}")
        sys.exit(0)

    launch_if_not_running("LM Studio.exe", cfg.get("lm_studio_path", ""))
    launch_if_not_running(
        "LTX Desktop.exe",
        cfg.get("ltx_desktop_path", ""),
        port=cfg.get("ltx_desktop_port"),
    )
    launch_if_not_running("StabilityMatrix.exe", cfg.get("stability_matrix_path", ""))

    result = subprocess.run([str(VENV_PYTHON), "app.py"])
    if result.returncode != 0:
        print()
        print("ERROR: Application exited with an error.")
        pause_exit(1)


if __name__ == "__main__":
    main()
