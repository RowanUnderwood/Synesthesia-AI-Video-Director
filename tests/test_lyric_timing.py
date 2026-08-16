import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lyric_timing import (align_caption_cues, align_lyric_lines,
                          alignment_needs_fallback, better_alignment,
                          load_valid_alignment, parse_timed_captions,
                          preferred_fallback_model, save_alignment,
                          strip_lyric_timestamps, transcribe_audio)


class FakeProject:
    def __init__(self, root):
        self.base_dir = root
        self.current_project = "project"
        self.project_dir = os.path.join(root, self.current_project)
        os.makedirs(os.path.join(self.project_dir, "assets"))

    def save_lyrics(self, text):
        with open(os.path.join(self.project_dir, "lyrics.txt"), "w", encoding="utf-8") as handle:
            handle.write(text)

    def get_lyrics(self):
        with open(os.path.join(self.project_dir, "lyrics.txt"), encoding="utf-8") as handle:
            return handle.read()

    def get_asset_path_if_exists(self, filename):
        path = os.path.join(self.project_dir, "assets", filename)
        return path if os.path.isfile(path) else None


class LyricTimingTests(unittest.TestCase):
    def test_monotonic_alignment_handles_repeated_chorus_lines(self):
        lyrics = "[Verse 1]\nHello, world\n[Chorus]\nWe rise\nWe rise"
        words = [
            {"word": "hello", "start": 1.0, "end": 1.4},
            {"word": "world", "start": 1.5, "end": 2.0},
            {"word": "we", "start": 5.0, "end": 5.2},
            {"word": "rise", "start": 5.3, "end": 5.8},
            {"word": "we", "start": 9.0, "end": 9.2},
            {"word": "rise", "start": 9.3, "end": 9.8},
        ]
        result = align_lyric_lines(lyrics, words)
        lines = result["timestamped_lyrics"].splitlines()
        self.assertEqual(result["matched_lines"], 3)
        self.assertTrue(lines[1].startswith("[00:01.000 --> 00:02.000]"))
        self.assertTrue(lines[3].startswith("[00:05.000 --> 00:05.800]"))
        self.assertTrue(lines[4].startswith("[00:09.000 --> 00:09.800]"))

    def test_unmatched_lines_remain_visibly_untimestamped(self):
        result = align_lyric_lines(
            "Known words here\nCompletely absent phrase",
            [
                {"word": "known", "start": 2.0, "end": 2.2},
                {"word": "words", "start": 2.3, "end": 2.5},
                {"word": "here", "start": 2.6, "end": 2.9},
            ],
        )
        lines = result["timestamped_lyrics"].splitlines()
        self.assertTrue(lines[0].startswith("[00:02.000"))
        self.assertEqual(lines[1], "Completely absent phrase")

    def test_scattered_words_cannot_create_an_impossible_line_span(self):
        lyric = "a million sisters of toil"
        words = [
            {"word": word, "start": index * 7.0, "end": index * 7.0 + 0.3,
             "probability": 0.99}
            for index, word in enumerate(lyric.split())
        ]
        result = align_lyric_lines(lyric, words, audio_duration=35.0)
        self.assertEqual(result["matched_lines"], 0)
        self.assertEqual(result["timestamped_lyrics"], lyric)

    def test_repeated_refrains_consume_local_occurrences_in_order(self):
        lyrics = "we rise\nwe rise\nwe rise"
        words = []
        for start in (5.0, 15.0, 25.0):
            words.extend([
                {"word": "we", "start": start, "end": start + 0.2, "probability": 0.95},
                {"word": "rise", "start": start + 0.3, "end": start + 0.8,
                 "probability": 0.95},
            ])
        result = align_lyric_lines(lyrics, words, audio_duration=30.0)
        lines = result["timestamped_lyrics"].splitlines()
        self.assertEqual(result["matched_lines"], 3)
        self.assertTrue(lines[0].startswith("[00:05.000"))
        self.assertTrue(lines[1].startswith("[00:15.000"))
        self.assertTrue(lines[2].startswith("[00:25.000"))

    def test_fallback_helpers_choose_stronger_coverage_and_english_model(self):
        weak = {"matched_lines": 2, "eligible_lines": 10, "coverage": 0.2,
                "mean_confidence": 0.9}
        strong = {"matched_lines": 7, "eligible_lines": 10, "coverage": 0.7,
                  "mean_confidence": 0.7}
        self.assertTrue(alignment_needs_fallback(weak))
        self.assertFalse(alignment_needs_fallback(strong))
        self.assertEqual(preferred_fallback_model("en"), "medium.en")
        self.assertIs(better_alignment((weak, {}), (strong, {}))[0], strong)

    def test_transcription_is_cpu_int8_without_ground_truth_decoder_prompt(self):
        calls = {}

        class FakeModel:
            def __init__(self, model_name, **kwargs):
                calls["model_name"] = model_name
                calls["init"] = kwargs

            def transcribe(self, audio_path, **kwargs):
                calls["audio_path"] = audio_path
                calls["transcribe"] = kwargs
                segment = SimpleNamespace(
                    start=1.0,
                    end=2.0,
                    text=" hello",
                    avg_logprob=-0.1,
                    no_speech_prob=0.01,
                    words=[SimpleNamespace(word=" hello", start=1.0, end=2.0,
                                           probability=0.9)],
                )
                info = SimpleNamespace(
                    language="en", language_probability=0.99, duration=3.0
                )
                return iter([segment]), info

        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = FakeModel
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result = transcribe_audio("vocals.mp3", "small")

        self.assertEqual(calls["init"], {"device": "cpu", "compute_type": "int8"})
        self.assertFalse(calls["transcribe"]["condition_on_previous_text"])
        self.assertNotIn("initial_prompt", calls["transcribe"])
        self.assertEqual(result["raw_text"], "hello")
        self.assertEqual(result["words"][0]["probability"], 0.9)

    def test_sbv_caption_import_matches_ground_truth_and_repeated_lines(self):
        sbv = """0:00:01.000,0:00:02.000
Hello world

0:00:05.000,0:00:06.000
We rise

0:00:09.000,0:00:10.000
We rise
"""
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "captions.sbv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(sbv)
            cues, caption_format = parse_timed_captions(path)
        result = align_caption_cues("[Verse]\nHello, world\n[Chorus]\nWe rise\nWe rise", cues)
        self.assertEqual(caption_format, "SBV")
        self.assertEqual(result["matched_lines"], 3)
        self.assertIn("[00:01.000 --> 00:02.000] Hello, world", result["timestamped_lyrics"])
        self.assertIn("[00:09.000 --> 00:10.000] We rise", result["timestamped_lyrics"])

    def test_srt_vtt_and_lrc_parsers(self):
        samples = {
            "captions.srt": "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            "captions.vtt": "WEBVTT\n\n00:01.000 --> 00:02.000\nHello\n",
            "captions.lrc": "[00:01.00]Hello\n[00:03.00]World\n",
        }
        with tempfile.TemporaryDirectory() as root:
            parsed = {}
            for filename, content in samples.items():
                path = os.path.join(root, filename)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                parsed[filename] = parse_timed_captions(path)
        self.assertEqual(parsed["captions.srt"][1], "SRT")
        self.assertEqual(parsed["captions.vtt"][1], "VTT")
        self.assertEqual(parsed["captions.lrc"][1], "LRC")
        self.assertEqual(parsed["captions.lrc"][0][0]["end"], 3.0)

    def test_timestamp_stripping_is_idempotent(self):
        timed = "[00:01.000 --> 00:02.000] Don't stop\n[Chorus]"
        self.assertEqual(strip_lyric_timestamps(timed), "Don't stop\n[Chorus]")

    def test_alignment_metadata_detects_lyrics_and_audio_changes(self):
        with tempfile.TemporaryDirectory() as root:
            pm = FakeProject(root)
            vocals = os.path.join(pm.project_dir, "assets", "vocals.mp3")
            with open(vocals, "wb") as handle:
                handle.write(b"vocals-one")
            alignment = {
                "raw_lyrics": "Hello world",
                "timestamped_lyrics": "[00:01.000 --> 00:02.000] Hello world",
                "matched_lines": 1,
                "eligible_lines": 1,
                "records": [],
            }
            save_alignment(
                pm, vocals, alignment, "en", transcription={"raw_text": "hello world"}
            )
            metadata, error = load_valid_alignment(pm)
            self.assertIsNone(error)
            self.assertEqual(metadata["matched_lines"], 1)
            self.assertEqual(metadata["version"], 2)
            self.assertEqual(metadata["transcription"]["raw_text"], "hello world")

            pm.save_lyrics("edited lyrics")
            self.assertIn("Lyrics changed", load_valid_alignment(pm)[1])
            pm.save_lyrics(alignment["timestamped_lyrics"])
            with open(vocals, "wb") as handle:
                handle.write(b"vocals-two")
            self.assertIn("vocals track changed", load_valid_alignment(pm)[1])


if __name__ == "__main__":
    unittest.main()
