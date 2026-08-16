import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import pandas as pd

import render_pipeline
import video


class FakePM:
    def __init__(self, root):
        self.base_dir = root
        self.current_project = "demo"
        os.makedirs(os.path.join(root, "demo", "videos"), exist_ok=True)
        self.df = pd.DataFrame([{
            "Shot_ID": "S001", "Type": "Action", "Duration": 3.0,
            "Total_Frames": 72, "Video_Prompt": "A test shot.", "Characters": "",
        }])
        self.queue_lock = threading.Lock()
        self.render_queue = []
        self.stop_video_generation = False
        self.character_reference_busy = False

    def get_path(self, subfolder):
        path = os.path.join(self.base_dir, self.current_project, subfolder)
        os.makedirs(path, exist_ok=True)
        return path

    def load_project_settings(self):
        return {}


def item():
    return {
        "shot_id": "S001", "resolution": "1080p",
        "vocal_mode": "Use Storyboard Prompt", "style": "None",
        "director": "None", "generation_mode": "Krea 2 First Frame",
        "use_llm_image_prompt": True, "caching_mode": "Use cached prompt",
    }


class RenderPipelineTests(unittest.TestCase):
    def test_enqueue_snapshots_h3_prompt_cache_policy(self):
        with tempfile.TemporaryDirectory() as root:
            pm = FakePM(root)
            queued = item()
            queued["h3_prompt_cache_mode"] = "Rewrite H3 prompts each render"
            pipeline = render_pipeline.H3RenderPipeline(pm, "demo")
            job_id = pipeline.enqueue(queued)
            self.assertEqual(
                pipeline.jobs[job_id].settings["h3_prompt_cache_mode"],
                "Rewrite H3 prompts each render",
            )

    def test_real_prompt_assembler_and_cached_frame_helper(self):
        with tempfile.TemporaryDirectory() as root:
            pm = FakePM(root)
            pm.character_bibles = {}
            row, row_index, prompt = video.assemble_shot_prompt(
                "S001", "Use Storyboard Prompt", pm,
            )
            self.assertEqual(row_index, 0)
            self.assertEqual(prompt, "A test shot.")
            frame_dir = pm.get_path("first_frames")
            frame_path = os.path.join(frame_dir, "S001.png")
            with open(frame_path, "wb") as handle:
                handle.write(b"frame")
            pm.df["First_Frame_Image_Path"] = os.path.relpath(
                frame_path, os.path.join(pm.base_dir, pm.current_project)
            )
            self.assertTrue(video._has_valid_cached_first_frame(
                {"shot_id": "S001", "caching_mode": "Use cached image"}, pm,
            ))

    def test_stages_complete_and_share_llm_limit_while_images_remain_serial(self):
        with tempfile.TemporaryDirectory() as root:
            pm = FakePM(root)
            llm_active = 0
            llm_peak = 0
            image_active = 0
            image_peak = 0
            counter_lock = threading.Lock()

            def llm_work(*_args, **_kwargs):
                nonlocal llm_active, llm_peak
                with counter_lock:
                    llm_active += 1
                    llm_peak = max(llm_peak, llm_active)
                time.sleep(0.03)
                with counter_lock:
                    llm_active -= 1

            def resolve(*_args, **_kwargs):
                llm_work()
                return "image prompt", ""

            def rewrite(*_args, **_kwargs):
                llm_work()
                return {"requires_target": True}, "h3 prompt"

            def image_work(*_args, **_kwargs):
                nonlocal image_active, image_peak
                with counter_lock:
                    image_active += 1
                    image_peak = max(image_peak, image_active)
                time.sleep(0.02)
                with counter_lock:
                    image_active -= 1
                return os.path.join(root, "frame.png")

            def fake_generate(*_args, **_kwargs):
                time.sleep(0.01)
                yield os.path.join(root, f"{threading.get_ident()}.mp4"), "Done"

            spec = {"requires_target": True, "mode": "FL2VA"}
            with mock.patch.object(render_pipeline, "assemble_shot_prompt",
                                   return_value=(pm.df.iloc[0], 0, "source")), \
                 mock.patch.object(render_pipeline, "h3_prompt_spec", return_value=spec), \
                 mock.patch.object(render_pipeline, "resolve_h3_target_prompt", side_effect=resolve), \
                 mock.patch.object(render_pipeline, "generate_prepared_h3_target_frame", side_effect=image_work), \
                 mock.patch.object(render_pipeline, "prepare_h3_rewrite", side_effect=rewrite), \
                 mock.patch.object(render_pipeline, "generate_h3_video_for_shot", side_effect=fake_generate):
                pipeline = render_pipeline.H3RenderPipeline(pm, "demo", llm_concurrency=2)
                pipeline.start([item(), item(), item(), item()])
                pipeline.thread.join(timeout=5)

            snap = pipeline.snapshot()
            self.assertEqual(snap["completed"], 4)
            self.assertEqual(snap["failed"], 0)
            self.assertLessEqual(llm_peak, 2)
            self.assertGreaterEqual(llm_peak, 2)
            self.assertEqual(image_peak, 1)

    def test_coop_is_blocked_when_available_commit_is_below_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            pm = FakePM(root)
            pipeline = render_pipeline.H3RenderPipeline(
                pm, "demo", coop_enabled=True, min_available_commit_gb=84,
            )
            pipeline.queues["video"].put(render_pipeline.PipelineJob(item()))
            pipeline.queues["video"].put(render_pipeline.PipelineJob(item()))
            with mock.patch.object(render_pipeline, "h3_instance_preflight", return_value=(True, "ok")), \
                 mock.patch.object(render_pipeline, "free_comfy_models", return_value=(True, "ok")), \
                 mock.patch.object(render_pipeline, "unload_lm_studio_models", return_value=(True, "ok")), \
                 mock.patch.object(render_pipeline, "system_commit_state",
                                   return_value={"used_gb": 120, "limit_gb": 180, "available_gb": 60}):
                pipeline._try_start_coop()
            self.assertEqual(pipeline.coop_state, "Blocked")
            self.assertIn("60.0 GB", pipeline.last_message)

    def test_coop_starts_only_after_successful_commit_preflight(self):
        class FakeThread:
            def __init__(self, *args, **kwargs):
                self.started = False

            def start(self):
                self.started = True

            def is_alive(self):
                return False

        with tempfile.TemporaryDirectory() as root:
            pm = FakePM(root)
            pipeline = render_pipeline.H3RenderPipeline(
                pm, "demo", coop_enabled=True, min_available_commit_gb=84,
            )
            pipeline.queues["video"].put(render_pipeline.PipelineJob(item()))
            pipeline.queues["video"].put(render_pipeline.PipelineJob(item()))
            with mock.patch.object(render_pipeline, "h3_instance_preflight", return_value=(True, "ok")), \
                 mock.patch.object(render_pipeline, "free_comfy_models", return_value=(True, "ok")), \
                 mock.patch.object(render_pipeline, "unload_lm_studio_models", return_value=(True, "ok")), \
                 mock.patch.object(render_pipeline, "system_commit_state",
                                   return_value={"used_gb": 80, "limit_gb": 180, "available_gb": 100}), \
                 mock.patch.object(render_pipeline.threading, "Thread", FakeThread):
                pipeline._try_start_coop()
            self.assertEqual(pipeline.coop_state, "Active")
            self.assertTrue(pipeline.coop_thread.started)


if __name__ == "__main__":
    unittest.main()
