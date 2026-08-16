"""Staged MiniMax H3 render queue with optional 4090 video co-op.

The Gradio UI remains synchronous, so this module owns daemon worker threads and
exposes thread-safe snapshots for polling. Blocking LM Studio and ComfyUI calls
never run in the UI event thread.
"""

from __future__ import annotations

import ctypes
import os
import queue
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

import config
import gpu_power
from models import LLMBridge
from h3 import (
    generate_h3_video_for_shot,
    generate_prepared_h3_target_frame,
    h3_prompt_spec,
    prepare_h3_rewrite,
    resolve_h3_target_prompt,
)
from video import assemble_shot_prompt


STAGES = ("image_prompt", "image", "h3_prompt", "video")
STAGE_LABELS = {
    "image_prompt": "Image prompts (LM Studio)",
    "image": "First frames (4090)",
    "h3_prompt": "H3 prompts (LM Studio)",
    "video": "H3 video (5090)",
}


@dataclass
class PipelineJob:
    item: dict
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_prompt: str = ""
    row_index: Optional[int] = None
    image_prompt: str = ""
    target_frame: Optional[str] = None
    h3_prompt: str = ""
    spec: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    llm_model: str = ""
    lm_studio_url: str = ""
    lm_studio_token: str = ""
    stage: str = "image_prompt"
    message: str = "Queued"
    output_path: Optional[str] = None
    error: str = ""

    @property
    def shot_id(self) -> str:
        return str(self.item.get("shot_id", "?"))


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong),
    ]


def system_commit_state() -> Optional[dict]:
    """Return Windows system commit totals in GiB, or None when unavailable."""
    if os.name != "nt":
        return None
    try:
        info = _PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        scale = info.PageSize / (1024 ** 3)
        total = info.CommitTotal * scale
        limit = info.CommitLimit * scale
        return {"used_gb": total, "limit_gb": limit, "available_gb": max(0.0, limit - total)}
    except Exception:
        return None


def free_comfy_models(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            f"{str(base_url).rstrip('/')}/free",
            json={"unload_models": True, "free_memory": True}, timeout=30,
        )
        response.raise_for_status()
        return True, "Image ComfyUI models unloaded"
    except Exception as exc:
        return False, f"Could not unload image ComfyUI models: {exc}"


def h3_instance_preflight(base_url: str) -> tuple[bool, str]:
    base = str(base_url).rstrip('/')
    required = ("MiniMaxH3Director", "MiniMaxH3ReferenceToVideo")
    try:
        # Modern ComfyUI supports a targeted route. It avoids transferring the
        # very large all-node schema, which can exceed 30 seconds on installs
        # with many custom-node packs.
        missing = []
        targeted_supported = True
        for name in required:
            response = requests.get(f"{base}/object_info/{name}", timeout=15)
            if response.status_code == 404:
                targeted_supported = False
                break
            response.raise_for_status()
            if name not in response.json():
                missing.append(name)
        if not targeted_supported:
            response = requests.get(f"{base}/object_info", timeout=90)
            response.raise_for_status()
            classes = response.json()
            missing = [name for name in required if name not in classes]
        if missing:
            return False, "The image ComfyUI instance is missing: " + ", ".join(missing)
        return True, "H3 nodes found on the image ComfyUI instance"
    except Exception as exc:
        return False, f"Image ComfyUI H3 preflight failed: {exc}"


def _lm_studio_is_local(lm_studio_url: str | None = None) -> bool:
    try:
        host = urlparse(lm_studio_url or config.LM_STUDIO_URL).hostname or ""
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True
        return host in set(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        return False


def unload_lm_studio_models(lm_studio_url: str | None = None) -> tuple[bool, str]:
    if not _lm_studio_is_local(lm_studio_url):
        return False, "LM Studio is remote; refusing to unload a different local instance."
    candidates = (
        os.path.expandvars(r"%USERPROFILE%\.cache\lm-studio\bin\lms.exe"),
        os.path.expandvars(r"%USERPROFILE%\.lmstudio\bin\lms.exe"),
        "lms",
    )
    for executable in candidates:
        if executable != "lms" and not os.path.isfile(executable):
            continue
        try:
            result = subprocess.run(
                [executable, "unload", "--all"], capture_output=True, text=True,
                timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            continue
        except Exception as exc:
            return False, f"LM Studio unload failed: {exc}"
        if result.returncode == 0:
            return True, "LM Studio models unloaded"
        detail = (result.stderr or result.stdout or "").strip().replace("\n", " ")
        return False, f"lms unload exited {result.returncode}: {detail[:200]}"
    return False, "lms CLI not found; co-op cannot safely free the LLM's system commit."


class H3RenderPipeline:
    """One project-owned staged H3 render run."""

    def __init__(self, pm, project: str, llm_concurrency: int = 4,
                 coop_enabled: bool = False, min_available_commit_gb: float = 84.0):
        self.pm = pm
        self.project = project
        self.llm_concurrency = max(1, min(4, int(llm_concurrency)))
        self.coop_enabled = bool(coop_enabled)
        self.min_available_commit_gb = max(8.0, float(min_available_commit_gb))
        self.image_comfy_url = config.COMFYUI_URL
        self.video_comfy_url = config.H3_COMFYUI_URL
        self.lm_studio_url = config.LM_STUDIO_URL
        self.lm_studio_token = config.LM_STUDIO_API_TOKEN
        self.llm_model = config.LM_STUDIO_MODEL
        self.queues = {stage: queue.Queue() for stage in STAGES}
        self.jobs: Dict[str, PipelineJob] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()
        self.coop_retire_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.worker_threads: list[threading.Thread] = []
        self.active_by_stage = {stage: set() for stage in STAGES}
        self.completed = 0
        self.failed = 0
        self.total = 0
        self.phase = "idle"
        self.last_message = ""
        self.coop_state = "Off"
        self.coop_thread: Optional[threading.Thread] = None
        self.completion_times: list[float] = []
        self.started_at = 0.0

    def is_active(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and not self.shutdown_event.is_set())

    def enqueue(self, item: dict) -> str:
        settings = dict(self.pm.load_project_settings())
        settings["h3_prompt_cache_mode"] = item.get(
            "h3_prompt_cache_mode",
            settings.get("h3_prompt_cache_mode", "Reuse cached H3 prompts"),
        )
        job = PipelineJob(
            dict(item), settings=settings,
            llm_model=self.llm_model, lm_studio_url=self.lm_studio_url,
            lm_studio_token=self.lm_studio_token,
        )
        with self.lock:
            self.jobs[job.job_id] = job
            self.total += 1
            self.queues["image_prompt"].put(job)
            if self.coop_state in ("Active", "Starting"):
                self.coop_retire_event.set()
                self.coop_state = "Retiring"
        return job.job_id

    def start(self, items: list[dict]) -> None:
        if self.is_active():
            for item in items:
                self.enqueue(item)
            return
        for item in items:
            self.enqueue(item)
        self.thread = threading.Thread(target=self._run, name="h3-pipeline", daemon=True)
        self.thread.start()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_event.set()
            self.phase = "paused"
        else:
            self.pause_event.clear()
            if self.is_active():
                self.phase = "running"

    def stop(self) -> None:
        self.phase = "stopping"
        self.stop_event.set()
        self.pause_event.clear()
        self.coop_retire_event.set()
        self.pm.stop_video_generation = True
        with self.lock:
            for q in self.queues.values():
                while True:
                    try:
                        job = q.get_nowait()
                        q.task_done()
                        if isinstance(job, PipelineJob) and not job.error:
                            job.error = "Cancelled before starting."
                            job.stage = "cancelled"
                            self.failed += 1
                    except queue.Empty:
                        break

    def cancel_pending(self) -> int:
        """Remove queued jobs while allowing currently active backend work to finish."""
        cancelled = 0
        with self.lock:
            for q in self.queues.values():
                while True:
                    try:
                        job = q.get_nowait()
                        q.task_done()
                        if isinstance(job, PipelineJob) and not job.error:
                            job.error = "Cancelled while pending."
                            job.stage = "cancelled"
                            self.failed += 1
                            cancelled += 1
                    except queue.Empty:
                        break
            if cancelled:
                self.last_message = f"🧹 Cancelled {cancelled} pending job(s)."
        return cancelled

    def snapshot(self) -> dict:
        with self.lock:
            active = {stage: sorted(values) for stage, values in self.active_by_stage.items()}
            pending = {stage: self.queues[stage].qsize() for stage in STAGES}
            done = self.completed + self.failed
            progress = int(100 * done / max(1, self.total))
            eta = None
            if len(self.completion_times) >= 2:
                intervals = [b - a for a, b in zip(self.completion_times, self.completion_times[1:])]
                eta = (sum(intervals) / len(intervals)) * max(0, self.total - done)
            return {
                "active": active, "pending": pending, "completed": self.completed,
                "failed": self.failed, "total": self.total, "progress": progress,
                "phase": self.phase, "message": self.last_message,
                "coop": self.coop_state, "eta_seconds": eta,
            }

    def status_text(self) -> str:
        snap = self.snapshot()
        lines = [
            f"🚦 PIPELINED H3 — {snap['phase'].upper()} — "
            f"{snap['completed']} done, {snap['failed']} failed, {snap['total']} total",
            f"🤝 4090 co-op: {snap['coop']}",
        ]
        for stage in STAGES:
            active = ", ".join(snap["active"][stage]) or "—"
            lines.append(
                f"{STAGE_LABELS[stage]}: {snap['pending'][stage]} queued; active {active}"
            )
        if snap["message"]:
            lines.append(snap["message"])
        return "\n".join(lines)

    def _wait_if_paused(self) -> bool:
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.2)
        return not self.stop_event.is_set()

    def _set_active(self, stage: str, job: PipelineJob, active: bool) -> None:
        with self.lock:
            label = f"{job.shot_id}:{job.job_id[:4]}"
            if active:
                self.active_by_stage[stage].add(label)
                job.stage = stage
            else:
                self.active_by_stage[stage].discard(label)

    def _fail(self, job: PipelineJob, exc: Exception) -> None:
        with self.lock:
            job.error = str(exc)
            job.message = f"❌ {job.shot_id}: {exc}"
            job.stage = "failed"
            self.failed += 1
            self.last_message = job.message
            self.completion_times = (self.completion_times + [time.time()])[-6:]

    def _image_prompt_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                job = self.queues["image_prompt"].get(timeout=0.25)
            except queue.Empty:
                continue
            self._set_active("image_prompt", job, True)
            try:
                if not self._wait_if_paused():
                    raise RuntimeError("Cancelled.")
                row, row_index, source = assemble_shot_prompt(
                    job.shot_id, job.item.get("vocal_mode"), self.pm,
                    job.item.get("style"), job.item.get("director"),
                )
                job.source_prompt, job.row_index = source, row_index
                spec = h3_prompt_spec(
                    job.shot_id, row, self.pm, source, job.item.get("vocal_mode"),
                    settings_override=job.settings,
                )
                job.spec = spec
                if spec["requires_target"]:
                    purpose = "h3_vocal_storyboard" if spec["mode"] == "LIPSYNC_TARGET" else "h3_action_target"
                    if job.item.get("use_llm_image_prompt"):
                        with self.llm_semaphore:
                            job.image_prompt, notice = resolve_h3_target_prompt(
                                job.shot_id, row, row_index, self.pm, source,
                                job.item.get("caching_mode", "Use cached prompt"), True, purpose,
                                settings_override=job.settings,
                                llm_bridge=LLMBridge(job.lm_studio_url, job.lm_studio_token),
                                llm_model=job.llm_model,
                            )
                    else:
                        job.image_prompt, notice = resolve_h3_target_prompt(
                            job.shot_id, row, row_index, self.pm, source,
                            job.item.get("caching_mode", "Use cached prompt"), False, purpose,
                            settings_override=job.settings,
                        )
                    job.message = notice or "Image prompt ready"
                    if self.stop_event.is_set():
                        raise RuntimeError("Cancelled.")
                    self.queues["image"].put(job)
                else:
                    if self.stop_event.is_set():
                        raise RuntimeError("Cancelled.")
                    self.queues["h3_prompt"].put(job)
            except Exception as exc:
                self._fail(job, exc)
            finally:
                self._set_active("image_prompt", job, False)
                self.queues["image_prompt"].task_done()

    def _image_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                job = self.queues["image"].get(timeout=0.25)
            except queue.Empty:
                continue
            self._set_active("image", job, True)
            try:
                if not self._wait_if_paused():
                    raise RuntimeError("Cancelled.")
                row = self.pm.df.loc[job.row_index]
                job.target_frame = generate_prepared_h3_target_frame(
                    job.shot_id, row, job.row_index, self.pm, job.image_prompt,
                    job.item.get("generation_mode", "Krea 2 First Frame"),
                    stop_check=self.stop_event.is_set,
                    settings_override=job.settings,
                )
                job.message = "First frame ready"
                if self.stop_event.is_set():
                    raise RuntimeError("Cancelled.")
                self.queues["h3_prompt"].put(job)
            except Exception as exc:
                self._fail(job, exc)
            finally:
                self._set_active("image", job, False)
                self.queues["image"].task_done()

    def _h3_prompt_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                job = self.queues["h3_prompt"].get(timeout=0.25)
            except queue.Empty:
                continue
            self._set_active("h3_prompt", job, True)
            try:
                if not self._wait_if_paused():
                    raise RuntimeError("Cancelled.")
                row = self.pm.df.loc[job.row_index]
                with self.llm_semaphore:
                    job.spec, job.h3_prompt = prepare_h3_rewrite(
                        job.shot_id, row, self.pm, job.source_prompt,
                        job.item.get("vocal_mode"),
                        settings_override=job.settings, llm_model=job.llm_model,
                        llm_bridge=LLMBridge(job.lm_studio_url, job.lm_studio_token),
                    )
                job.message = "H3 prompt ready"
                if self.stop_event.is_set():
                    raise RuntimeError("Cancelled.")
                self.queues["video"].put(job)
            except Exception as exc:
                self._fail(job, exc)
            finally:
                self._set_active("h3_prompt", job, False)
                self.queues["h3_prompt"].task_done()

    def _video_worker(self, base_url: str, coop: bool = False) -> None:
        while not self.shutdown_event.is_set():
            if coop and self.coop_retire_event.is_set():
                break
            try:
                job = self.queues["video"].get(timeout=0.25)
            except queue.Empty:
                continue
            self._set_active("video", job, True)
            try:
                if not self._wait_if_paused():
                    raise RuntimeError("Cancelled.")
                prepared = {
                    "target_frame": job.target_frame,
                    "target_prompt": job.image_prompt,
                    "h3_prompt": job.h3_prompt,
                    "settings": job.settings,
                }
                output = None
                row = self.pm.df.loc[job.row_index]
                for path, message in generate_h3_video_for_shot(
                    job.shot_id, row, job.row_index, self.pm, job.source_prompt,
                    generation_mode=job.item.get("generation_mode", "Krea 2 First Frame"),
                    use_llm_image_prompt=False,
                    caching_mode=job.item.get("caching_mode", "Use cached prompt"),
                    vocal_mode=job.item.get("vocal_mode"),
                    h3_prompt_cache_mode=job.settings.get(
                        "h3_prompt_cache_mode", "Reuse cached H3 prompts"
                    ),
                    prepared=prepared, comfyui_url=base_url,
                ):
                    job.message = message or job.message
                    if path:
                        output = path
                if not output:
                    raise RuntimeError(job.message or "H3 returned no video.")
                with self.lock:
                    job.output_path = output
                    job.stage = "complete"
                    self.completed += 1
                    self.last_message = f"✅ {job.shot_id} completed on {'4090' if coop else '5090'}"
                    self.completion_times = (self.completion_times + [time.time()])[-6:]
            except Exception as exc:
                self._fail(job, exc)
            finally:
                self._set_active("video", job, False)
                self.queues["video"].task_done()
        if coop:
            with self.lock:
                self.coop_state = "Off"
                self.completion_times.clear()

    def _upstream_busy(self) -> bool:
        with self.lock:
            return any(self.active_by_stage[stage] for stage in STAGES[:-1]) or any(
                not self.queues[stage].empty() for stage in STAGES[:-1]
            )

    def _try_start_coop(self) -> None:
        if not self.coop_enabled or self.coop_state != "Off" or self._upstream_busy():
            return
        if self.queues["video"].qsize() < 2 or self.pm.character_reference_busy:
            return
        if self.coop_thread and self.coop_thread.is_alive():
            return
        self.coop_state = "Starting"
        ok, message = h3_instance_preflight(self.image_comfy_url)
        if not ok:
            self.coop_state, self.last_message = "Blocked", f"🤝 {message}"
            return
        ok, message = free_comfy_models(self.image_comfy_url)
        if not ok:
            self.coop_state, self.last_message = "Blocked", f"🤝 {message}"
            return
        ok, message = unload_lm_studio_models(self.lm_studio_url)
        if not ok:
            self.coop_state, self.last_message = "Blocked", f"🤝 {message}"
            return
        commit = system_commit_state()
        if commit is None:
            self.coop_state, self.last_message = "Blocked", "🤝 Windows system commit could not be measured."
            return
        if commit["available_gb"] < self.min_available_commit_gb:
            self.coop_state = "Blocked"
            self.last_message = (
                f"🤝 Co-op blocked: {commit['available_gb']:.1f} GB commit available; "
                f"requires {self.min_available_commit_gb:.1f} GB."
            )
            return
        # Service calls above can take more than a minute. Re-check the gate so
        # a newly injected image/prompt job or a drained video queue cannot race
        # us into starting the second renderer with the wrong resource role.
        with self.pm.queue_lock:
            injected_waiting = bool(self.pm.render_queue)
        if (self.stop_event.is_set() or injected_waiting or self._upstream_busy()
                or self.queues["video"].qsize() < 2):
            self.coop_state = "Off"
            self.last_message = "🤝 Co-op transition deferred because queue state changed during preflight."
            return
        self.coop_retire_event.clear()
        self.coop_state = "Active"
        self.last_message = (
            f"🤝 4090 co-op engaged with {commit['available_gb']:.1f} GB system commit available."
        )
        self.completion_times.clear()
        self.coop_thread = threading.Thread(
            target=self._video_worker, args=(self.image_comfy_url, True),
            name="h3-video-4090", daemon=True,
        )
        self.coop_thread.start()

    def _run(self) -> None:
        self.phase = "running"
        self.started_at = time.time()
        self.pm.stop_video_generation = False
        if config.POWER_LIMIT_MODE == "wattage_cap":
            _ok, self.last_message = gpu_power.apply_limits(gpu_power.watts_from_settings(
                config.get_machine_settings()
            ))
        self.llm_semaphore = threading.Semaphore(self.llm_concurrency)
        self.worker_threads = [
            *[threading.Thread(target=self._image_prompt_worker, daemon=True,
                               name=f"h3-image-prompt-{index}")
              for index in range(self.llm_concurrency)],
            threading.Thread(target=self._image_worker, daemon=True, name="h3-image-4090"),
            *[threading.Thread(target=self._h3_prompt_worker, daemon=True,
                               name=f"h3-prompt-{index}")
              for index in range(self.llm_concurrency)],
            threading.Thread(target=self._video_worker, args=(self.video_comfy_url, False),
                             daemon=True, name="h3-video-5090"),
        ]
        for worker in self.worker_threads:
            worker.start()
        idle_ticks = 0
        try:
            while not self.stop_event.is_set():
                # Absorb jobs added through the existing queue controls mid-run.
                with self.pm.queue_lock:
                    injected = list(self.pm.render_queue)
                    self.pm.render_queue.clear()
                for item in injected:
                    self.enqueue(item)
                if injected and self.coop_state in ("Active", "Starting"):
                    self.coop_retire_event.set()
                    self.coop_state = "Retiring"
                self._try_start_coop()
                snap = self.snapshot()
                active = any(snap["active"][stage] for stage in STAGES)
                queued = any(snap["pending"][stage] for stage in STAGES)
                if not active and not queued and snap["completed"] + snap["failed"] >= snap["total"]:
                    idle_ticks += 1
                    if idle_ticks >= 3:
                        break
                else:
                    idle_ticks = 0
                time.sleep(0.5)
        finally:
            self.shutdown_event.set()
            self.coop_retire_event.set()
            for worker in self.worker_threads:
                worker.join(timeout=5)
            if self.coop_thread:
                self.coop_thread.join(timeout=5)
            self.pm.stop_video_generation = False
            self.phase = "stopped" if self.stop_event.is_set() else "complete"
