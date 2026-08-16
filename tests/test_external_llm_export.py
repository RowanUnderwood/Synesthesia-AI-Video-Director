import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd

import config
from external_llm_export import ExternalExportError, create_external_llm_bundle
from lyric_timing import save_alignment


class FakeProject:
    def __init__(self, root, with_timeline=True):
        self.base_dir = root
        self.current_project = "project"
        self.project_dir = os.path.join(root, self.current_project)
        os.makedirs(os.path.join(self.project_dir, "assets"))
        self.character_bibles = {}
        self.df = pd.DataFrame([{
            "Shot_ID": "S001", "Type": "Vocal", "Start_Time": 0.0,
            "End_Time": 5.0, "Duration": 5.0, "Start_Frame": 0,
            "End_Frame": 120, "Total_Frames": 120, "Video_Prompt": "",
        }]) if with_timeline else pd.DataFrame()

    def save_lyrics(self, text):
        with open(os.path.join(self.project_dir, "lyrics.txt"), "w", encoding="utf-8") as handle:
            handle.write(text)

    def get_lyrics(self):
        path = os.path.join(self.project_dir, "lyrics.txt")
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def get_asset_path_if_exists(self, filename):
        path = os.path.join(self.project_dir, "assets", filename)
        return path if os.path.isfile(path) else None

    def load_project_settings(self):
        return {
            "video_mode": "Intercut", "h3_lead_character": "Alice",
            "h3_aspect": "16:9 - Landscape", "vocal_prompt_mode": "Use Storyboard Prompt",
        }


class ExternalLLMExportTests(unittest.TestCase):
    def _aligned_project(self, root):
        pm = FakeProject(root)
        vocals = os.path.join(pm.project_dir, "assets", "vocals.mp3")
        with open(vocals, "wb") as handle:
            handle.write(b"isolated vocals")
        save_alignment(pm, vocals, {
            "raw_lyrics": "Matched line\nUnmatched line",
            "timestamped_lyrics": "[00:01.000 --> 00:02.000] Matched line\nUnmatched line",
            "matched_lines": 1,
            "eligible_lines": 2,
            "records": [],
        }, "en")
        return pm

    def test_bundle_contains_three_files_and_blank_bible_header(self):
        with tempfile.TemporaryDirectory() as root, patch.object(config, "VIDEO_BACKEND", "MiniMax H3"):
            pm = self._aligned_project(root)
            status, bundle = create_external_llm_bundle(pm)
            self.assertIn("1 lyric line", status)
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "shot_list.csv", "character_bibles.csv", "external_llm_instructions.txt",
                })
                self.assertEqual(
                    archive.read("character_bibles.csv").decode("utf-8").strip(),
                    "character_name,description",
                )
                instructions = archive.read("external_llm_instructions.txt").decode("utf-8")
                self.assertIn("Include the lead singer in character_bibles.csv", instructions)
                self.assertIn("Matched 1 of 2 lyric lines", instructions)

    def test_export_rejects_missing_timeline(self):
        with tempfile.TemporaryDirectory() as root:
            pm = FakeProject(root, with_timeline=False)
            with self.assertRaisesRegex(ExternalExportError, "Build the timeline"):
                create_external_llm_bundle(pm)

    def test_ltx_bundle_does_not_require_singer_bible_entry(self):
        with tempfile.TemporaryDirectory() as root, patch.object(config, "VIDEO_BACKEND", "LTX Desktop"):
            pm = self._aligned_project(root)
            _, bundle = create_external_llm_bundle(pm)
            with zipfile.ZipFile(bundle) as archive:
                instructions = archive.read("external_llm_instructions.txt").decode("utf-8")
            self.assertIn("may remain in the separate singer/performance description", instructions)
            self.assertNotIn("H3 requires that Character Bible entry", instructions)


if __name__ == "__main__":
    unittest.main()
