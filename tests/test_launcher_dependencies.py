import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import launcher


class LauncherDependencyTests(unittest.TestCase):
    def _paths(self, root):
        root = Path(root)
        requirements = root / "requirements.txt"
        requirements.write_text("faster-whisper>=1.1,<2.0\n", encoding="utf-8")
        venv = root / "venv"
        venv.mkdir()
        return requirements, venv / ".requirements.sha256", venv / "Scripts" / "python.exe"

    def test_changed_requirements_are_installed_and_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            requirements, marker, python = self._paths(root)
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch.object(launcher, "REQUIREMENTS_FILE", requirements),
                patch.object(launcher, "REQUIREMENTS_MARKER", marker),
                patch.object(launcher, "VENV_PYTHON", python),
                patch.object(launcher, "_run", return_value=completed) as run,
            ):
                self.assertTrue(launcher.ensure_dependencies())

            run.assert_called_once_with(
                str(python), "-m", "pip", "install", "-r", str(requirements),
            )
            self.assertEqual(
                marker.read_text(encoding="ascii").strip(),
                hashlib.sha256(requirements.read_bytes()).hexdigest(),
            )

    def test_unchanged_requirements_skip_pip(self):
        with tempfile.TemporaryDirectory() as root:
            requirements, marker, python = self._paths(root)
            marker.write_text(
                hashlib.sha256(requirements.read_bytes()).hexdigest() + "\n",
                encoding="ascii",
            )
            with (
                patch.object(launcher, "REQUIREMENTS_FILE", requirements),
                patch.object(launcher, "REQUIREMENTS_MARKER", marker),
                patch.object(launcher, "VENV_PYTHON", python),
                patch.object(launcher, "_run") as run,
            ):
                self.assertTrue(launcher.ensure_dependencies())
                run.assert_not_called()

    def test_forced_update_retains_upgrade_behavior(self):
        with tempfile.TemporaryDirectory() as root:
            requirements, marker, python = self._paths(root)
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch.object(launcher, "REQUIREMENTS_FILE", requirements),
                patch.object(launcher, "REQUIREMENTS_MARKER", marker),
                patch.object(launcher, "VENV_PYTHON", python),
                patch.object(launcher, "_run", return_value=completed) as run,
            ):
                self.assertTrue(launcher.ensure_dependencies(force=True, fatal=False))
            run.assert_called_once_with(
                str(python), "-m", "pip", "install", "--upgrade",
                "-r", str(requirements),
            )

    def test_failed_update_does_not_record_requirements(self):
        with tempfile.TemporaryDirectory() as root:
            requirements, marker, python = self._paths(root)
            completed = subprocess.CompletedProcess([], 1)
            with (
                patch.object(launcher, "REQUIREMENTS_FILE", requirements),
                patch.object(launcher, "REQUIREMENTS_MARKER", marker),
                patch.object(launcher, "VENV_PYTHON", python),
                patch.object(launcher, "_run", return_value=completed),
            ):
                self.assertFalse(launcher.ensure_dependencies(fatal=False))
            self.assertFalse(marker.exists())

    def test_backend_setting_is_read_without_importing_app_config(self):
        with tempfile.TemporaryDirectory() as root:
            settings = Path(root) / "global_settings.json"
            settings.write_text('{"video_backend": "MiniMax H3"}', encoding="utf-8")
            self.assertEqual(launcher.load_video_backend(settings), "MiniMax H3")

    def test_missing_or_invalid_backend_falls_back_to_ltx(self):
        with tempfile.TemporaryDirectory() as root:
            settings = Path(root) / "global_settings.json"
            self.assertEqual(launcher.load_video_backend(settings), "LTX Desktop")
            settings.write_text('{"video_backend": "Unknown"}', encoding="utf-8")
            self.assertEqual(launcher.load_video_backend(settings), "LTX Desktop")

    def test_minimax_launches_h3_comfy_and_skips_ltx(self):
        cfg = {
            "lm_studio_path": "lm.exe", "ltx_desktop_path": "ltx.exe",
            "comfy_image_launcher_path": "image.bat",
            "comfy_video_launcher_path": "video.bat",
        }
        with patch.object(launcher, "launch_if_not_running") as launch:
            launcher.launch_backend_services(cfg, "MiniMax H3")
        calls = launch.call_args_list
        self.assertIn(call("cmd.exe", "image.bat", port=8188), calls)
        self.assertIn(call("cmd.exe", "video.bat", port=8189), calls)
        self.assertFalse(any(call.args and call.args[0] == "LTX Desktop.exe" for call in calls))

    def test_ltx_launches_desktop_and_skips_h3_comfy(self):
        cfg = {
            "lm_studio_path": "lm.exe", "ltx_desktop_path": "ltx.exe",
            "ltx_desktop_port": 8000,
            "comfy_image_launcher_path": "image.bat",
            "comfy_video_launcher_path": "video.bat",
        }
        with patch.object(launcher, "launch_if_not_running") as launch:
            launcher.launch_backend_services(cfg, "LTX Desktop")
        calls = launch.call_args_list
        self.assertIn(call("LTX Desktop.exe", "ltx.exe", port=8000), calls)
        self.assertNotIn(call("cmd.exe", "video.bat", port=8189), calls)


if __name__ == "__main__":
    unittest.main()
