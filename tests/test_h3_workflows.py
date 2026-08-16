import json
import hashlib
import os
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

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
        output = workflow["2568"]["inputs"]
        self.assertEqual(output["codec"], "H.264")
        self.assertEqual(output["container"], "MP4")
        self.assertEqual(output["bit_depth"], "8-bit")
        self.assertEqual(output["audio_codec"], "AAC")

    def test_custom_aspect_uses_the_workflow_enum_and_dimensions(self):
        workflow = h3.patch_h3_fl2(
            "frame.png", "prompt", 5, "synesthesia_h3/test", "CUSTOM",
            "0.65 MP - Balanced", 21, 9, seed=1,
        )
        inputs = workflow["1512:2531"]["inputs"]
        self.assertEqual(inputs["aspect_preset_when_not_image"], "CUSTOM")
        self.assertEqual((inputs["custom_aspect_width"], inputs["custom_aspect_height"]), (21, 9))
        output = workflow["2568"]["inputs"]
        self.assertEqual((output["codec"], output["container"]), ("H.264", "MP4"))

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
        self.assertEqual(workflow["100"]["inputs"]["aspect_ratio"], "16:9 (Widescreen)")
        self.assertNotIn("ref_images.ref_image_2", workflow["110"]["inputs"])
        self.assertFalse(workflow["2293"]["inputs"]["trim_to_audio"])
        self.assertFalse(workflow["2402"]["inputs"]["trim_to_audio"])

    def test_lipsync_patch_adds_target_as_picture_one_and_honors_4_3(self):
        workflow = h3.patch_h3_lipsync(
            "face.png", "body.png", "shot.wav", "prompt", 8.0,
            "synesthesia_h3/test", seed=5, target_image="setting.png",
            aspect="4:3 - Landscape",
        )
        self.assertEqual(workflow["912"]["inputs"]["image"], "setting.png")
        inputs = workflow["110"]["inputs"]
        self.assertEqual(inputs["ref_images.ref_image_0"], ["912", 0])
        self.assertEqual(inputs["ref_images.ref_image_1"], ["910", 0])
        self.assertEqual(inputs["ref_images.ref_image_2"], ["911", 0])
        self.assertEqual(workflow["100"]["inputs"]["aspect_ratio"], "4:3 (Standard)")

    def test_lipsync_patch_uses_explicit_dimensions_for_custom_ratio(self):
        workflow = h3.patch_h3_lipsync(
            "face.png", "body.png", "shot.wav", "prompt", 8.0,
            "synesthesia_h3/test", seed=6, aspect="CUSTOM",
            custom_width=5, custom_height=8,
        )
        self.assertEqual(workflow["100"]["class_type"], "PrimitiveInt")
        self.assertEqual(workflow["103"]["class_type"], "PrimitiveInt")
        width = workflow["100"]["inputs"]["value"]
        height = workflow["103"]["inputs"]["value"]
        self.assertEqual((width % 32, height % 32), (0, 0))
        self.assertAlmostEqual(width / height, 5 / 8, delta=0.06)
        self.assertEqual(workflow["110"]["inputs"]["height"], ["103", 0])

    def test_lipsync_target_rewrite_contract_assigns_picture_roles(self):
        class FakePM:
            def load_project_settings(self):
                return {}

            def save_project_settings(self, value):
                self.saved = value

        with patch.object(h3.LLMBridge, "query", return_value="rewritten") as query:
            result = h3.rewrite_h3_prompt(
                FakePM(), "Singer performs in a station.", 5.0, "LIPSYNC_TARGET",
                ["<Picture 1>: target frame", "<Picture 2>: face", "<Picture 3>: body"],
                "vision-model",
            )
        self.assertEqual(result, "rewritten")
        system_prompt = query.call_args.args[0]
        self.assertIn("<Picture 1> as a standalone concrete first frame", system_prompt)
        self.assertIn("<Picture 2> for facial identity", system_prompt)
        self.assertIn("<Picture 3> for full-body", system_prompt)
        self.assertIn("never invent lyrics", system_prompt)

    def test_h3_prompt_cache_can_be_reused_or_explicitly_bypassed(self):
        class FakePM:
            queue_lock = threading.Lock()

            def __init__(self):
                self.settings = {}

            def load_project_settings(self):
                return dict(self.settings)

            def save_project_settings(self, value):
                self.settings.update(value)

        pm = FakePM()
        bridge = Mock()
        bridge.query.side_effect = ["first rewrite", "fresh rewrite"]
        args = (pm, "A figure crosses a bridge.", 5.0, "FL2VA",
                ["<Picture 1>: starting frame"], "vision-model")

        self.assertEqual(h3.rewrite_h3_prompt(*args, llm_bridge=bridge), "first rewrite")
        self.assertEqual(h3.rewrite_h3_prompt(*args, llm_bridge=bridge), "first rewrite")
        self.assertEqual(bridge.query.call_count, 1)

        self.assertEqual(
            h3.rewrite_h3_prompt(*args, llm_bridge=bridge, use_cache=False),
            "fresh rewrite",
        )
        self.assertEqual(bridge.query.call_count, 2)
        self.assertEqual(h3.rewrite_h3_prompt(*args, llm_bridge=bridge), "fresh rewrite")

    def test_h3_prompt_cache_is_bounded(self):
        class FakePM:
            queue_lock = threading.Lock()

            def __init__(self):
                self.settings = {
                    "h3_prompt_cache": {
                        f"old-{index}": f"prompt-{index}"
                        for index in range(h3.H3_PROMPT_CACHE_LIMIT)
                    }
                }

            def load_project_settings(self):
                return dict(self.settings)

            def save_project_settings(self, value):
                self.settings.update(value)

        pm = FakePM()
        bridge = Mock()
        bridge.query.return_value = "new rewrite"
        h3.rewrite_h3_prompt(
            pm, "New shot", 4.0, "FL2VA", ["<Picture 1>: starting frame"],
            "vision-model", llm_bridge=bridge,
        )
        cache = pm.settings["h3_prompt_cache"]
        self.assertEqual(len(cache), h3.H3_PROMPT_CACHE_LIMIT)
        self.assertNotIn("old-0", cache)
        self.assertIn("new rewrite", cache.values())

    def test_storyboard_vocal_routes_target_frame_into_lipsync_patch(self):
        with tempfile.TemporaryDirectory() as root:
            setting = os.path.join(root, "setting.png")
            face = os.path.join(root, "face.png")
            body = os.path.join(root, "body.png")
            for path in (setting, face, body):
                with open(path, "wb") as handle:
                    handle.write(b"reference")

            class FakePM:
                current_project = "project"
                base_dir = root
                queue_lock = threading.Lock()
                stop_video_generation = False
                df = pd.DataFrame([{
                    "Shot_ID": "S001", "Type": "Vocal", "Duration": 5.0,
                    "Total_Frames": 120, "Status": "Pending", "Video_Path": "",
                    "Render_Resolution": "",
                }])

                def load_project_settings(self):
                    return {
                        "h3_lead_character": "Alice", "h3_aspect": "4:3 - Landscape",
                        "h3_quality": "0.65 MP - Balanced", "h3_custom_width": 16,
                        "h3_custom_height": 9, "h3_lipsync_output": "Native",
                    }

                def get_path(self, name):
                    path = os.path.join(root, name)
                    os.makedirs(path, exist_ok=True)
                    return path

                def save_data(self):
                    pass

            class FakeClient:
                def __init__(self, _url):
                    pass

                def upload_input(self, path, _job_id):
                    return os.path.basename(path)

                def submit(self, _workflow, _client_id):
                    return "prompt-id"

                def download(self, _descriptor, destination):
                    return str(destination)

            pm = FakePM()
            with patch.object(h3, "ComfyClient", FakeClient), \
                    patch.object(h3, "h3_reference_paths", return_value=(face, body)), \
                    patch.object(h3, "_generate_h3_target_frame", return_value=[
                        (setting, {"prompt": "setting prompt"}),
                    ]) as target_generator, \
                    patch.object(h3, "rewrite_h3_prompt", return_value="rewritten") as rewrite, \
                    patch.object(h3, "_h3_audio_chunk", return_value=os.path.join(root, "shot.wav")), \
                    patch.object(h3, "patch_h3_lipsync", wraps=h3.patch_h3_lipsync) as builder, \
                    patch.object(h3, "_record_job"), \
                    patch.object(h3, "_wait_for_output", return_value={"filename": "shot.mp4"}):
                updates = list(h3.generate_h3_video_for_shot(
                    "S001", pm.df.loc[0], 0, pm, "Storyboard venue prompt",
                    vocal_mode="Use Storyboard Prompt",
                ))

            self.assertTrue(target_generator.called)
            self.assertEqual(rewrite.call_args.args[3], "LIPSYNC_TARGET")
            self.assertEqual(builder.call_args.kwargs["target_image"], "setting.png")
            self.assertEqual(builder.call_args.kwargs["aspect"], "4:3 - Landscape")
            self.assertTrue(updates[-1][0].endswith(".mp4"))


if __name__ == "__main__":
    unittest.main()
