import json
import hashlib
import os
import tempfile
import threading
import unittest

import h3
import pandas as pd


class H3WorkflowTests(unittest.TestCase):
    def test_grid_padding_covers_timeline_without_changing_it(self):
        self.assertEqual(h3.h3_render_frames(360), 362)
        self.assertEqual(h3.h3_render_frames(361), 362)
        self.assertEqual(h3.h3_render_frames(362), 362)
        with self.assertRaises(ValueError):
            h3.h3_render_frames(363)

    def test_ref2_patch_keeps_all_uploaded_images_in_order(self):
        workflow = h3.patch_h3_ref2(
            ["face_alex.png", "body_alex.png", "face_jo.png", "body_jo.png"],
            "full reference prompt", 5, "synesthesia_h3/test",
            "3:4 - Photo", "0.65 MP - Balanced", 16, 9, seed=99,
        )
        director = workflow["2693"]["inputs"]
        timeline = json.loads(director["timeline_data"])
        self.assertEqual(director["mode"], "REF2VA")
        self.assertEqual(director["prompt"], "full reference prompt")
        self.assertEqual([item["value"] for item in timeline["items"]], [
            "face_alex.png", "body_alex.png", "face_jo.png", "body_jo.png",
        ])
        self.assertEqual(workflow["1512:2590"]["inputs"]["steps"], 4)
        self.assertIn("ref2v_turbo_4step", workflow["2678"]["inputs"]["stack_data"])

    def test_custom_aspect_uses_the_workflow_enum_and_dimensions(self):
        workflow = h3.patch_h3_fl2(
            "frame.png", "prompt", 5, "synesthesia_h3/test", "CUSTOM",
            "0.65 MP - Balanced", 21, 9, seed=1,
        )
        inputs = workflow["1512:2531"]["inputs"]
        self.assertEqual(inputs["aspect_preset_when_not_image"], "CUSTOM")
        self.assertEqual((inputs["custom_aspect_width"], inputs["custom_aspect_height"]), (21, 9))

    def test_landscape_aspects_use_custom_ratio_and_krea_documented_geometry(self):
        for label, ratio, dimensions in (
            ("16:9 - Landscape", (16, 9), (1376, 768)),
            ("4:3 - Landscape", (4, 3), (1184, 896)),
        ):
            with self.subTest(label=label):
                workflow = h3.patch_h3_fl2(
                    "frame.png", "prompt", 5, "synesthesia_h3/test", label,
                    "0.65 MP - Balanced", 1, 1, seed=1,
                )
                inputs = workflow["1512:2531"]["inputs"]
                self.assertEqual(inputs["aspect_preset_when_not_image"], "CUSTOM")
                self.assertEqual(
                    (inputs["custom_aspect_width"], inputs["custom_aspect_height"]), ratio
                )
                self.assertEqual(h3.h3_target_dimensions(label), dimensions)

    def test_ref2_target_frame_plus_four_characters_fills_nine_slots(self):
        images = ["setting.png"] + [f"character_{index}.png" for index in range(8)]
        workflow = h3.patch_h3_ref2(
            images, "reference prompt", 5, "synesthesia_h3/test",
            "16:9 - Landscape", "0.65 MP - Balanced", 16, 9, seed=2,
        )
        timeline = json.loads(workflow["2693"]["inputs"]["timeline_data"])
        self.assertEqual([item["value"] for item in timeline["items"]], images)

    def test_portrait_character_reference_uses_unoriented_wlsh_ratio(self):
        workflow = h3.patch_krea2(
            "full-body reference", "synesthesia_h3/body", "4:3", "portrait", 896, seed=3
        )
        inputs = workflow["15"]["inputs"]
        self.assertEqual(inputs["aspect"], "4:3")
        self.assertEqual(inputs["direction"], "portrait")
        self.assertEqual(inputs["shortside"], 896)

    def test_character_reference_is_invalidated_when_description_changes(self):
        with tempfile.TemporaryDirectory() as root:
            face = os.path.join(root, "face.png")
            body = os.path.join(root, "body.png")
            for path in (face, body):
                with open(path, "wb") as handle:
                    handle.write(b"reference")

            original = "Silver hair and a red coat."

            class FakePM:
                character_bibles = {"Alice": original}

                def load_project_settings(self):
                    return {"h3_character_references": {"Alice": {
                        "face_path": face,
                        "body_path": body,
                        "description_hash": hashlib.sha256(original.encode()).hexdigest(),
                    }}}

            pm = FakePM()
            self.assertEqual(h3.h3_reference_paths(pm, "Alice"), (face, body))
            pm.character_bibles["Alice"] = "Black hair and a blue coat."
            self.assertEqual(h3.h3_reference_paths(pm, "Alice"), (None, None))

    def test_target_frame_cache_requires_matching_aspect_prompt_and_generator(self):
        prompt = "A cathedral-like computation hall at sunrise."
        with tempfile.TemporaryDirectory() as root:
            project_root = os.path.join(root, "project")
            frames = os.path.join(project_root, "first_frames")
            os.makedirs(frames)
            frame_path = os.path.join(frames, "cached.png")
            with open(frame_path, "wb") as handle:
                handle.write(b"cached")

            class FakePM:
                base_dir = root
                current_project = "project"
                queue_lock = threading.Lock()
                df = pd.DataFrame([{
                    "First_Frame_Prompt": prompt,
                    "First_Frame_Image_Path": os.path.relpath(frame_path, project_root),
                    "First_Frame_Image_Source": "Krea 2",
                    "First_Frame_Image_Aspect": "16:9",
                    "First_Frame_Image_Prompt_Hash": hashlib.sha256(prompt.encode()).hexdigest(),
                }])

                def save_data(self):
                    pass

                def load_project_settings(self):
                    return {}

            updates = list(h3._generate_h3_target_frame(
                "S001", FakePM.df.loc[0], 0, FakePM(), "unused source",
                "Krea 2 First Frame", "Use cached image", "16:9 - Landscape",
                16, 9, True,
            ))
            self.assertEqual(updates[-1][0], frame_path)
            self.assertIn("Using cached Krea 2 target frame", updates[0][1])

    def test_lipsync_patch_uses_both_references_and_keeps_video_untrimmed(self):
        workflow = h3.patch_h3_lipsync(
            "face.png", "body.png", "shot.wav", "prompt", 15.0,
            "synesthesia_h3/test", seed=4,
        )
        self.assertEqual(workflow["910"]["inputs"]["image"], "face.png")
        self.assertEqual(workflow["911"]["inputs"]["image"], "body.png")
        self.assertEqual(workflow["940"]["inputs"]["audio"], "shot.wav")
        self.assertFalse(workflow["2293"]["inputs"]["trim_to_audio"])
        self.assertFalse(workflow["2402"]["inputs"]["trim_to_audio"])


if __name__ == "__main__":
    unittest.main()
