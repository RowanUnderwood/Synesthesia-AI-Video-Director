import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
