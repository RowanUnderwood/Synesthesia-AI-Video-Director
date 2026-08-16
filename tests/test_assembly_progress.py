import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from assembly import AssemblyProgressLogger, prepare_assembly_clip


class AssemblyProgressTests(unittest.TestCase):
    def test_legacy_webm_is_converted_once_and_cached(self):
        class FakePM:
            def __init__(self, root):
                self.root = root

            def get_path(self, name):
                return os.path.join(self.root, name)

        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "legacy.webm")
            with open(source, "wb") as handle:
                handle.write(b"legacy video")
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                with open(command[-1], "wb") as handle:
                    handle.write(b"x" * 2048)
                return subprocess.CompletedProcess(command, 0, "", "")

            updates = []
            with patch("assembly.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"), \
                    patch("assembly.subprocess.run", side_effect=fake_run):
                proxy = prepare_assembly_clip(
                    source, FakePM(root),
                    lambda fraction, message: updates.append((fraction, message)),
                    label="S001",
                )
                cached = prepare_assembly_clip(source, FakePM(root), label="S001")

            self.assertEqual(proxy, cached)
            self.assertTrue(proxy.endswith("_h264.mp4"))
            self.assertEqual(len(calls), 1)
            self.assertIn("av1_cuvid", calls[0][0])
            self.assertIn("h264_nvenc", calls[0][0])
            self.assertTrue(any("proxy ready" in message.lower() for _, message in updates))

    def test_proxy_conversion_falls_back_after_hardware_timeout(self):
        class FakePM:
            def __init__(self, root):
                self.root = root

            def get_path(self, name):
                return os.path.join(self.root, name)

        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "legacy.webm")
            with open(source, "wb") as handle:
                handle.write(b"legacy video")
            call_count = 0

            def fake_run(command, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                with open(command[-1], "wb") as handle:
                    handle.write(b"x" * 2048)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("assembly.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"), \
                    patch("assembly.subprocess.run", side_effect=fake_run):
                proxy = prepare_assembly_clip(source, FakePM(root), timeout=1)

            self.assertTrue(os.path.isfile(proxy))
            self.assertEqual(call_count, 2)

    def test_moviepy_video_frames_are_reported_as_encode_progress(self):
        updates = []
        logger = AssemblyProgressLogger(lambda fraction, message: updates.append((fraction, message)))
        logger(t__total=200)
        logger(t__index=50)

        fraction, message = updates[-1]
        self.assertAlmostEqual(fraction, 0.45 + 0.54 * 0.25)
        self.assertIn("frame 50/200", message)
        self.assertIn("25%", message)

    def test_moviepy_audio_chunks_are_reported_separately(self):
        updates = []
        logger = AssemblyProgressLogger(lambda fraction, message: updates.append((fraction, message)))
        logger(chunk__total=20)
        logger(chunk__index=10)

        fraction, message = updates[-1]
        self.assertAlmostEqual(fraction, 0.38 + 0.07 * 0.5)
        self.assertIn("audio", message.lower())
        self.assertIn("chunk 10/20", message)


if __name__ == "__main__":
    unittest.main()
