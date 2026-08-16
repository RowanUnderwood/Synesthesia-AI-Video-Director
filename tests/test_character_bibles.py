import os
import tempfile
import unittest

import pandas as pd

from models import ProjectManager, normalise_character_bible_table


class CharacterBibleTests(unittest.TestCase):
    def test_blank_add_row_is_ignored_when_saving(self):
        value = pd.DataFrame([
            {"character_name": "Alice", "description": "Red coat and silver hair."},
            {"character_name": "", "description": ""},
        ])
        bibles, clean = normalise_character_bible_table(value)
        self.assertEqual(bibles, {"Alice": "Red coat and silver hair."})
        self.assertEqual(clean.to_dict("records"), [{
            "character_name": "Alice", "description": "Red coat and silver hair."
        }])

    def test_incomplete_character_is_rejected(self):
        value = pd.DataFrame([{"character_name": "Alice", "description": ""}])
        with self.assertRaisesRegex(ValueError, "needs a description"):
            normalise_character_bible_table(value)

    def test_case_insensitive_duplicate_is_rejected(self):
        value = pd.DataFrame([
            {"character_name": "Alice", "description": "First"},
            {"character_name": "alice", "description": "Second"},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate character name"):
            normalise_character_bible_table(value)

    def test_replace_persists_and_only_assigns_names_already_in_prompts(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "legacy"))
            pm = ProjectManager.__new__(ProjectManager)
            pm.base_dir = root
            pm.current_project = "legacy"
            pm.character_bibles = {}
            pm.df = pd.DataFrame([
                {"Video_Prompt": "Alice walks through the station."},
                {"Video_Prompt": "An empty station at night."},
            ])

            pm.replace_character_bibles(pd.DataFrame([
                {"character_name": "Alice", "description": "Silver hair and a red coat."},
                {"character_name": "Singer", "description": "A singer in a black suit."},
            ]))

            self.assertTrue(os.path.isfile(os.path.join(root, "legacy", "character_bibles.csv")))
            self.assertEqual(pm.df["Characters"].tolist(), ["Alice", ""])


if __name__ == "__main__":
    unittest.main()
