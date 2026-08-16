"""MiniMax H3 and Krea 2 ComfyUI integration.

The workflow exports in the repository are immutable templates.  This module
deep-copies a template for every job, records the submitted graph in the
project, and only patches documented inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Iterable

import requests
from pydub import AudioSegment

import config
from models import LLMBridge


H3_FPS = 24
H3_MIN_FRAMES = 5
H3_FRAME_STEP = 17
H3_MAX_FRAMES = 362
H3_MAX_CHARACTERS_PER_REF2V_SHOT = 4  # one target frame + two identity references each = nine images

KREA_WORKFLOW = "krea2_native_workflow_Jakes version for silly hat.json"
H3_FL2_WORKFLOW = "DasiwaMinimaxH3WorkflowsT2VA_cMMH3V10_jakes version_API.json"
H3_REF2_WORKFLOW = "DasiwaMinimaxH3Workflows REF2VA API.json"
H3_LIPSYNC_WORKFLOW = "H3 Single Shot - Lip-Sync + Reference images_API.json"

H3_ASPECT_PRESETS = [
    "16:9 - Landscape", "4:3 - Landscape",
    "1:1 - Square", "2:3 - Classic", "3:4 - Photo", "5:8 - Tall",
    "9:16 - Social", "9:21 - Cinema", "CUSTOM",
]

H3_LANDSCAPE_ASPECTS = {
    "16:9 - Landscape": {
        "ratio": "16:9", "width": 1376, "height": 768,
        "krea_aspect": "16:9", "krea_direction": "landscape", "krea_shortside": 768,
    },
    "4:3 - Landscape": {
        "ratio": "4:3", "width": 1184, "height": 896,
        "krea_aspect": "4:3", "krea_direction": "landscape", "krea_shortside": 896,
    },
}
H3_QUALITY_PRESETS = [
    "144p", "240p", "360p", "480p", "540p", "576p", "720p", "900p",
    "1024p", "1080p", "1152p", "1440p", "2160p", "2K", "4K",
    "0.26 MP - Preview", "0.36 MP - Small", "0.52 MP - SD",
    "0.65 MP - Balanced", "0.83 MP - HD", "1.00 MP - 1024p",
    "1.05 MP - HD+", "1.20 MP - HD++", "1.35 MP - 2K lite",
    "1.55 MP - 2K", "1.65 MP - 2K+", "1.75 MP - QHD",
    "2.10 MP - FHD", "3.30 MP - QHD+", "4.75 MP - 2K Pro",
    "6.50 MP - Production", "8.30 MP - UHD",
]


class ComfyError(RuntimeError):
    """A ComfyUI request or workflow failed."""


def _root_path(filename: str) -> Path:
    return Path(__file__).resolve().with_name(filename)


def h3_render_frames(timeline_frames: int) -> int:
    """Return the smallest H3-supported render length covering timeline_frames.

    The returned extra frames are *render padding*, never timeline padding.  The
    assembly keeps the original timeline frame count, which prevents cumulative
    audio drift over a long project.
    """
    target = max(H3_MIN_FRAMES, int(timeline_frames))
    frames = target + ((H3_MIN_FRAMES - target) % H3_FRAME_STEP)
    if frames > H3_MAX_FRAMES:
        raise ValueError(
            f"H3 supports at most {H3_MAX_FRAMES} frames "
            f"({H3_MAX_FRAMES / H3_FPS:.3f}s); this shot needs {target} frames."
        )
    return frames


def h3_render_duration(timeline_frames: int) -> float:
    return h3_render_frames(timeline_frames) / H3_FPS


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return slug or "item"


def _workflow_template(filename: str, expected_nodes: dict[str, str]) -> dict:
    path = _root_path(filename)
    try:
        with path.open("r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    except Exception as exc:
        raise ComfyError(f"Could not load {filename}: {exc}") from exc
    if not isinstance(workflow, dict) or "nodes" in workflow:
        raise ComfyError(f"{filename} must be a ComfyUI API-format workflow.")
    for node_id, expected in expected_nodes.items():
        actual = workflow.get(node_id, {}).get("class_type")
        if actual != expected:
            raise ComfyError(
                f"{filename} node {node_id}: expected {expected!r}, got {actual!r}."
            )
    return workflow


class ComfyClient:
    """Small synchronous ComfyUI client with job-scoped uploads and output retrieval."""

    def __init__(self, base_url: str):
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url:
            raise ComfyError("ComfyUI URL is empty.")
        self.session = requests.Session()

    def upload_input(self, source: str | Path, job_id: str) -> str:
        source = Path(source)
        if not source.is_file():
            raise ComfyError(f"Input file does not exist: {source}")
        upload_name = f"synesthesia_{_safe_slug(job_id)}_{uuid.uuid4().hex[:8]}_{source.name}"
        mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        try:
            with source.open("rb") as handle:
                response = self.session.post(
                    f"{self.base_url}/upload/image",
                    files={"image": (upload_name, handle, mime)},
                    data={"type": "input", "overwrite": "true"},
                    timeout=60,
                )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            detail = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
            raise ComfyError(f"ComfyUI upload failed: {exc} {detail}") from exc
        name = str(data.get("name") or "").strip()
        if not name:
            raise ComfyError(f"ComfyUI upload returned no filename: {data}")
        subfolder = str(data.get("subfolder") or "").strip("/\\")
        return "/".join(part for part in (subfolder, name) if part)

    def submit(self, workflow: dict, client_id: str) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            detail = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
            raise ComfyError(f"ComfyUI submission failed: {exc} {detail}") from exc
        if data.get("error"):
            raise ComfyError(str(data.get("node_errors") or data["error"]))
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI returned no prompt_id: {data}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict | None:
        try:
            response = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
            response.raise_for_status()
            return response.json().get(prompt_id)
        except requests.RequestException as exc:
            raise ComfyError(f"Could not read ComfyUI history for {prompt_id}: {exc}") from exc

    def cancel_owned(self, prompt_id: str) -> None:
        """Cancel a job without interrupting somebody else's work on a shared server."""
        try:
            queue = self.session.get(f"{self.base_url}/queue", timeout=15).json()
            self.session.post(f"{self.base_url}/queue", json={"delete": [prompt_id]}, timeout=15)
            # /interrupt is instance-wide, so use it only after confirming the
            # active item belongs to this application-owned prompt id.
            if str(prompt_id) in json.dumps(queue.get("queue_running", [])):
                self.session.post(f"{self.base_url}/interrupt", timeout=15)
        except requests.RequestException:
            pass

    def download(self, descriptor: dict, destination: str | Path) -> str:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self.session.get(
                f"{self.base_url}/view",
                params={
                    "filename": descriptor["filename"],
                    "subfolder": descriptor.get("subfolder", ""),
                    "type": descriptor.get("type", "output"),
                },
                timeout=600,
                stream=True,
            )
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        except requests.RequestException as exc:
            raise ComfyError(f"Could not download ComfyUI output: {exc}") from exc
        return str(destination)


def _find_descriptor(value, allowed_extensions: set[str]) -> dict | None:
    if isinstance(value, dict):
        if "filename" in value:
            suffix = Path(str(value["filename"])).suffix.lower()
            if not allowed_extensions or suffix in allowed_extensions:
                return value
        for nested in value.values():
            result = _find_descriptor(nested, allowed_extensions)
            if result:
                return result
    elif isinstance(value, list):
        for nested in value:
            result = _find_descriptor(nested, allowed_extensions)
            if result:
                return result
    return None


def _wait_for_output(client: ComfyClient, prompt_id: str, preferred_nodes: Iterable[str],
                     allowed_extensions: set[str], timeout: int = 1800, stop_check=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop_check and stop_check():
            client.cancel_owned(prompt_id)
            raise ComfyError("H3 render cancelled.")
        history = client.history(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise ComfyError(f"ComfyUI workflow failed: {status.get('messages') or status}")
            outputs = history.get("outputs", {})
            for node_id in preferred_nodes:
                descriptor = _find_descriptor(outputs.get(str(node_id)), allowed_extensions)
                if descriptor:
                    return descriptor
            descriptor = _find_descriptor(outputs, allowed_extensions)
            if descriptor:
                return descriptor
        time.sleep(1)
    raise ComfyError(f"Timed out waiting for ComfyUI job {prompt_id}.")


def _record_job(pm, job_id: str, workflow: dict, details: dict) -> None:
    try:
        directory = Path(pm.get_path("h3_jobs"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{job_id}.json").write_text(
            json.dumps({"details": details, "workflow": workflow}, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(f"Could not save H3 job manifest: {exc}")


def _patch_resolution(workflow: dict, aspect: str, quality: str,
                      custom_width: int = 16, custom_height: int = 9) -> None:
    resolution = workflow["1512:2531"]["inputs"]
    resolution["resolution_preset"] = quality
    resolution["scale_from_image"] = False
    if aspect in H3_LANDSCAPE_ASPECTS:
        spec = H3_LANDSCAPE_ASPECTS[aspect]
        resolution["aspect_preset_when_not_image"] = "CUSTOM"
        resolution["custom_aspect_width"] = int(spec["ratio"].split(":")[0])
        resolution["custom_aspect_height"] = int(spec["ratio"].split(":")[1])
        resolution["swap_aspect_when_not_image"] = False
    elif aspect == "CUSTOM":
        resolution["aspect_preset_when_not_image"] = "CUSTOM"
        resolution["custom_aspect_width"] = int(custom_width)
        resolution["custom_aspect_height"] = int(custom_height)
        resolution["swap_aspect_when_not_image"] = False
    else:
        resolution["aspect_preset_when_not_image"] = aspect


def h3_aspect_cache_key(aspect: str, custom_width: int = 16, custom_height: int = 9) -> str:
    if aspect in H3_LANDSCAPE_ASPECTS:
        return H3_LANDSCAPE_ASPECTS[aspect]["ratio"]
    if aspect == "CUSTOM":
        return f"{int(custom_width)}:{int(custom_height)}"
    return str(aspect)


def h3_krea_geometry(aspect: str, custom_width: int = 16, custom_height: int = 9) -> dict:
    if aspect in H3_LANDSCAPE_ASPECTS:
        return dict(H3_LANDSCAPE_ASPECTS[aspect])
    mappings = {
        "1:1 - Square": ("1:1", "landscape", 1024),
        "2:3 - Classic": ("3:2", "portrait", 832),
        "3:4 - Photo": ("4:3", "portrait", 896),
        "5:8 - Tall": ("5:8", "portrait", 768),
        "9:16 - Social": ("16:9", "portrait", 768),
        "9:21 - Cinema": ("21:9", "portrait", 672),
    }
    if aspect == "CUSTOM":
        width, height = int(custom_width), int(custom_height)
        direction = "landscape" if width >= height else "portrait"
        ratio = f"{max(width, height)}:{min(width, height)}" if direction == "portrait" else f"{width}:{height}"
        return {"ratio": f"{width}:{height}", "krea_aspect": ratio,
                "krea_direction": direction, "krea_shortside": 768}
    ratio, direction, shortside = mappings.get(aspect, ("16:9", "landscape", 768))
    return {"ratio": h3_aspect_cache_key(aspect, custom_width, custom_height),
            "krea_aspect": ratio, "krea_direction": direction, "krea_shortside": shortside}


def h3_target_dimensions(aspect: str, custom_width: int = 16, custom_height: int = 9) -> tuple[int, int]:
    spec = h3_krea_geometry(aspect, custom_width, custom_height)
    if spec.get("width") and spec.get("height"):
        return int(spec["width"]), int(spec["height"])
    raw = h3_aspect_cache_key(aspect, custom_width, custom_height)
    match = re.search(r"(\d+)\s*:\s*(\d+)", raw)
    width_ratio, height_ratio = (int(match.group(1)), int(match.group(2))) if match else (16, 9)
    shortside = int(spec.get("krea_shortside", 768))
    if width_ratio >= height_ratio:
        width, height = round(shortside * width_ratio / height_ratio), shortside
    else:
        width, height = shortside, round(shortside * height_ratio / width_ratio)
    # Both local image workflows are happiest on dimensions divisible by 32.
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


def patch_krea2(prompt: str, filename_prefix: str, aspect: str = "1:1",
                direction: str = "landscape", shortside: int = 1024, seed: int | None = None) -> dict:
    workflow = _workflow_template(KREA_WORKFLOW, {
        "2": "KSampler", "5": "PreviewImage", "6": "CLIPTextEncode",
        "15": "Empty Latent by Ratio (WLSH)",
    })
    workflow = copy.deepcopy(workflow)
    workflow["6"]["inputs"]["text"] = str(prompt)
    workflow["2"]["inputs"]["seed"] = int(seed if seed is not None else uuid.uuid4().int % 10**15)
    workflow["15"]["inputs"].update({
        "aspect": aspect, "direction": direction, "shortside": int(shortside), "batch_size": 1,
    })
    workflow["5"]["class_type"] = "SaveImage"
    workflow["5"]["inputs"]["filename_prefix"] = filename_prefix
    return workflow


def patch_h3_fl2(image_name: str, prompt: str, duration_seconds: float, filename_prefix: str,
                 aspect: str, quality: str, custom_width: int, custom_height: int,
                 seed: int | None = None) -> dict:
    workflow = _workflow_template(H3_FL2_WORKFLOW, {
        "2693": "MiniMaxH3Director", "2568": "DaSiWa_EnhancedVideoCombine",
        "1512:2531": "DaSiWa_ResolutionScaleCalculator", "1512:2600": "RandomNoise",
    })
    workflow = copy.deepcopy(workflow)
    item_id = f"image-{uuid.uuid4().hex}"
    timeline = {
        "version": 1,
        "items": [{"id": item_id, "enabled": True, "order": 0, "slot": 0, "start": 0,
                   "duration": 1, "type": "image", "value": image_name, "prompt": prompt}],
        "prompt_blocks": [{"id": f"attached-{item_id}", "text": prompt, "enabled": True,
                           "start": 0, "duration": 1, "order": 0}],
    }
    director = workflow["2693"]["inputs"]
    director.update({"mode": "FL2VA", "duration": float(duration_seconds), "prompt": "",
                     "timeline_data": json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))})
    _patch_resolution(workflow, aspect, quality, custom_width, custom_height)
    workflow["1512:2600"]["inputs"]["noise_seed"] = int(seed if seed is not None else uuid.uuid4().int % 10**15)
    workflow["2568"]["inputs"].update({"filename_prefix": filename_prefix, "save_output": True})
    return workflow


def patch_h3_ref2(image_names: list[str], prompt: str, duration_seconds: float, filename_prefix: str,
                  aspect: str, quality: str, custom_width: int, custom_height: int,
                  seed: int | None = None) -> dict:
    if not image_names or len(image_names) > 9:
        raise ValueError("Ref2VA accepts one to nine reference images.")
    workflow = _workflow_template(H3_REF2_WORKFLOW, {
        "2693": "MiniMaxH3Director", "2678": "DaSiWa_LTX2LoraLoader",
        "2568": "DaSiWa_EnhancedVideoCombine", "1512:2531": "DaSiWa_ResolutionScaleCalculator",
    })
    workflow = copy.deepcopy(workflow)
    items = [
        {"id": f"image-{uuid.uuid4().hex}", "enabled": True, "order": index, "slot": index,
         "start": index, "duration": 1, "type": "image", "value": name}
        for index, name in enumerate(image_names)
    ]
    timeline = {"version": 1, "items": items, "prompt_blocks": []}
    director = workflow["2693"]["inputs"]
    director.update({"mode": "REF2VA", "duration": float(duration_seconds), "prompt": prompt,
                     "timeline_data": json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))})
    _patch_resolution(workflow, aspect, quality, custom_width, custom_height)
    workflow["1512:2600"]["inputs"]["noise_seed"] = int(seed if seed is not None else uuid.uuid4().int % 10**15)
    workflow["2568"]["inputs"].update({"filename_prefix": filename_prefix, "save_output": True})
    return workflow


def patch_h3_lipsync(face_image: str, body_image: str, audio_name: str, prompt: str,
                     duration_seconds: float, filename_prefix: str, seed: int | None = None,
                     upscaled: bool = True) -> dict:
    workflow = _workflow_template(H3_LIPSYNC_WORKFLOW, {
        "110": "MiniMaxH3ReferenceToVideo", "910": "LoadImage", "911": "LoadImage",
        "940": "LoadAudio", "120": "RandomNoise", "2293": "VHS_VideoCombine",
        "2402": "VHS_VideoCombine",
    })
    workflow = copy.deepcopy(workflow)
    workflow["910"]["inputs"]["image"] = face_image
    workflow["911"]["inputs"]["image"] = body_image
    workflow["940"]["inputs"]["audio"] = audio_name
    workflow["110"]["inputs"]["prompt"] = prompt
    workflow["120"]["inputs"]["noise_seed"] = int(seed if seed is not None else uuid.uuid4().int % 10**15)
    workflow["101"]["inputs"]["value"] = float(duration_seconds)
    workflow["2293"]["inputs"].update({"filename_prefix": f"{filename_prefix}_native", "trim_to_audio": False})
    workflow["2402"]["inputs"].update({"filename_prefix": f"{filename_prefix}_rtx", "trim_to_audio": False})
    if not upscaled:
        # The output node remains present for workflow validity; callers select node 2293.
        workflow["2402"]["inputs"]["save_output"] = False
    return workflow


def _project_reference_settings(pm) -> dict:
    settings = pm.load_project_settings()
    refs = settings.get("h3_character_references", {})
    return refs if isinstance(refs, dict) else {}


def h3_reference_paths(pm, character_name: str) -> tuple[str | None, str | None]:
    record = _project_reference_settings(pm).get(character_name, {})
    description = str(pm.character_bibles.get(character_name, "")).strip()
    expected_hash = hashlib.sha256(description.encode()).hexdigest() if description else None
    if not isinstance(record, dict) or record.get("description_hash") != expected_hash:
        return None, None
    face = record.get("face_path") if isinstance(record, dict) else None
    body = record.get("body_path") if isinstance(record, dict) else None
    return (face if face and os.path.isfile(face) else None,
            body if body and os.path.isfile(body) else None)


def h3_reference_gallery(pm):
    gallery = []
    for name in pm.character_bibles:
        face, body = h3_reference_paths(pm, name)
        if face:
            gallery.append((face, f"{name} — face reference"))
        if body:
            gallery.append((body, f"{name} — full-body reference"))
    return gallery


def _run_image_workflow(client: ComfyClient, workflow: dict, pm, job_id: str, destination: Path) -> str:
    client_id = uuid.uuid4().hex
    prompt_id = client.submit(workflow, client_id)
    _record_job(pm, job_id, workflow, {"prompt_id": prompt_id, "workflow": "Krea 2"})
    descriptor = _wait_for_output(client, prompt_id, ["5"], {".png", ".jpg", ".jpeg", ".webp"}, timeout=900)
    return client.download(descriptor, destination)


def generate_h3_character_references(pm, character_names: Iterable[str] | None = None) -> list[str]:
    """Generate project-owned face and portrait references through Krea 2."""
    if not pm.current_project:
        raise ComfyError("Load a project before generating character references.")
    names = list(character_names or pm.character_bibles.keys())
    if not names:
        raise ComfyError("There are no character bibles to turn into references.")
    client = ComfyClient(config.COMFYUI_URL)
    directory = Path(pm.get_path("h3_references"))
    directory.mkdir(parents=True, exist_ok=True)
    records = _project_reference_settings(pm)
    completed = []
    for name in names:
        description = str(pm.character_bibles.get(name, "")).strip()
        if not description:
            raise ComfyError(f"Character bible for {name} is empty.")
        slug, token = _safe_slug(name), uuid.uuid4().hex[:8]
        face_prompt = (
            f"Photorealistic character identity reference, head-and-shoulders closeup of {name}. "
            f"{description}. Centered face, direct natural gaze, even studio light, clean neutral background, "
            "sharp facial detail, no text, no collage."
        )
        body_prompt = (
            f"Photorealistic full-body character identity and wardrobe reference of {name}. "
            f"{description}. Standing naturally, entire body and footwear visible, centered, neutral studio "
            "background, clear wardrobe detail, no text, no collage."
        )
        face = _run_image_workflow(
            client, patch_krea2(face_prompt, f"synesthesia_h3/{slug}_face_{token}", "1:1", "landscape", 1024),
            pm, f"{slug}_face_{token}", directory / f"{slug}_face.png",
        )
        body = _run_image_workflow(
            client, patch_krea2(body_prompt, f"synesthesia_h3/{slug}_body_{token}", "4:3", "portrait", 896),
            pm, f"{slug}_body_{token}", directory / f"{slug}_body.png",
        )
        records[name] = {
            "face_path": face, "body_path": body, "description_hash": hashlib.sha256(description.encode()).hexdigest(),
            "generated_at": time.time(), "face_prompt": face_prompt, "body_prompt": body_prompt,
        }
        completed.append(name)
    pm.save_project_settings({"h3_character_references": records})
    return completed


def _h3_prompt_cache_key(mode: str, prompt: str, duration: float, labels: list[str]) -> str:
    raw = json.dumps({"mode": mode, "prompt": prompt, "duration": duration, "labels": labels}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rewrite_h3_prompt(pm, source_prompt: str, duration: float, mode: str, labels: list[str],
                      llm_model: str) -> str:
    """Rewrite an existing storyboard prompt into the H3 contract and cache it per project."""
    key = _h3_prompt_cache_key(mode, source_prompt, duration, labels)
    settings = pm.load_project_settings()
    cache = settings.get("h3_prompt_cache", {})
    if isinstance(cache, dict) and cache.get(key):
        return str(cache[key])
    if mode == "REF2VA":
        system = (
            "You format MiniMax H3 full-reference video prompts. Return only six English sections in this "
            "exact order: subject_definitions, summary, retention_analysis, detailed_description, "
            "overall_soundscape, non_diegetic_music. Use the supplied <Picture N> labels consistently. "
            "Preserve any lyrics/dialogue verbatim and never invent <Audio> labels. When <Picture 1> is "
            "identified as a concrete target first frame, define it as a standalone picture entry, preserve "
            "its setting, viewpoint, composition, subject placement, lighting, and shot planning, and classify "
            "the task as [keyframe completion + reference generation]. Character face/body pictures are identity "
            "sources: define each character as one <Subject N> sourced from its two pictures rather than creating "
            "standalone picture entries for those identity images."
        )
        contract = "Use the full-reference Ref2VA format."
    else:
        system = (
            "You format MiniMax H3 FL2VA video prompts. Return only the required first alignment line, a blank "
            "line, then integrated_multimodal_description, overall_soundscape, and non_diegetic_music. "
            "Use <Picture 1> at 0.00 seconds and make all timing fit the requested duration exactly."
        )
        contract = "Use the base FL2VA format with one starting reference picture."
    label_text = "\n".join(f"- {label}" for label in labels) or "- <Picture 1>: starting image"
    user = (
        f"{contract}\nDuration: {duration:.3f} seconds\nReference labels:\n{label_text}\n\n"
        f"Storyboard prompt to rewrite:\n{source_prompt}"
    )
    rewritten = LLMBridge().query(system, user, llm_model, temperature=0.3)
    if not rewritten or rewritten.startswith("Error"):
        raise ComfyError(f"H3 prompt rewrite failed: {rewritten}")
    cache = cache if isinstance(cache, dict) else {}
    cache[key] = rewritten
    pm.save_project_settings({"h3_prompt_cache": cache})
    return rewritten


def _h3_audio_chunk(pm, row, render_frames: int, shot_id: str) -> str:
    source = pm.get_asset_path_if_exists("vocals.mp3") or pm.get_asset_path_if_exists("full_song.mp3")
    if not source:
        raise ComfyError("Missing vocals or full-song audio for H3 lip-sync.")
    start_frame = int(row.get("Start_Frame", round(float(row["Start_Time"]) * H3_FPS)))
    start_ms = round(start_frame * 1000 / H3_FPS)
    end_ms = round((start_frame + render_frames) * 1000 / H3_FPS)
    audio = AudioSegment.from_file(source)
    chunk = audio[start_ms:end_ms]
    required_ms = end_ms - start_ms
    if len(chunk) < required_ms:
        chunk += AudioSegment.silent(duration=required_ms - len(chunk), frame_rate=audio.frame_rate)
    path = Path(pm.get_path("audio_chunks")) / f"{_safe_slug(shot_id)}_h3_{uuid.uuid4().hex[:8]}.wav"
    chunk.export(path, format="wav")
    return str(path)


def _shot_character_names(row, pm) -> list[str]:
    raw = str(row.get("Characters", "") or "")
    names = [piece.strip() for piece in raw.split(",") if piece.strip()]
    valid = {name.casefold(): name for name in pm.character_bibles}
    unique = []
    for name in names:
        canonical = valid.get(name.casefold())
        if canonical and canonical not in unique:
            unique.append(canonical)
    return unique


def _clean_cell(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def _generate_h3_target_frame(shot_id, row, row_index: int, pm, source_prompt: str,
                              generation_mode: str, caching_mode: str, aspect: str,
                              custom_width: int, custom_height: int,
                              use_llm_image_prompt: bool):
    """Yield progress plus a final ``(path, {prompt: ...})`` record."""
    skip_prompt_cache = caching_mode == "Regenerate both on each render"
    first_frame_prompt = "" if skip_prompt_cache else _clean_cell(row.get("First_Frame_Prompt", ""))
    if not first_frame_prompt:
        if use_llm_image_prompt:
            yield None, "🧠 Creating the setting/target-frame prompt..."
            from video import convert_prompt_for_zimage
            first_frame_prompt = convert_prompt_for_zimage(source_prompt, pm, pm.load_project_settings())
            if not skip_prompt_cache:
                with pm.queue_lock:
                    pm.df.at[row_index, "First_Frame_Prompt"] = first_frame_prompt
                    pm.save_data()
        else:
            first_frame_prompt = source_prompt

    mode = generation_mode if generation_mode in ("Krea 2 First Frame", "Z-Image First Frame") else "Krea 2 First Frame"
    generator_name = "Krea 2" if mode == "Krea 2 First Frame" else "Z-Image (ComfyUI)"
    aspect_key = h3_aspect_cache_key(aspect, custom_width, custom_height)
    prompt_hash = hashlib.sha256(first_frame_prompt.encode("utf-8")).hexdigest()

    cached_rel = _clean_cell(row.get("First_Frame_Image_Path", ""))
    cached_path = cached_rel if os.path.isabs(cached_rel) else os.path.join(
        pm.base_dir, pm.current_project, cached_rel
    )
    cache_matches = (
        caching_mode == "Use cached image"
        and cached_rel and os.path.isfile(cached_path)
        and _clean_cell(row.get("First_Frame_Image_Source", "")) == generator_name
        and _clean_cell(row.get("First_Frame_Image_Aspect", "")) == aspect_key
        and _clean_cell(row.get("First_Frame_Image_Prompt_Hash", "")) == prompt_hash
    )
    if cache_matches:
        yield None, f"♻️ Using cached {generator_name} target frame ({aspect_key})..."
        yield cached_path, {"prompt": first_frame_prompt}
        return

    if cached_rel:
        with pm.queue_lock:
            for column in ("First_Frame_Image_Path", "First_Frame_Image_Source",
                           "First_Frame_Image_Aspect", "First_Frame_Image_Prompt_Hash"):
                pm.df.at[row_index, column] = ""
            pm.save_data()
        yield None, "♻️ Cached target frame did not match the selected aspect, prompt, or generator; regenerating..."

    frame_dir = Path(pm.get_path("first_frames"))
    frame_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    if mode == "Krea 2 First Frame":
        geometry = h3_krea_geometry(aspect, custom_width, custom_height)
        yield None, f"🖼️ Generating Krea 2 setting frame ({aspect_key})..."
        first_frame = _run_image_workflow(
            ComfyClient(config.COMFYUI_URL),
            patch_krea2(
                first_frame_prompt, f"synesthesia_h3/{_safe_slug(shot_id)}_{token}_frame",
                geometry["krea_aspect"], geometry["krea_direction"], geometry["krea_shortside"],
            ),
            pm, f"{_safe_slug(shot_id)}_{token}_frame",
            frame_dir / f"{_safe_slug(shot_id)}_h3_first_frame_{token}.png",
        )
    else:
        from video import generate_comfyui_zimage_first_frame
        width, height = h3_target_dimensions(aspect, custom_width, custom_height)
        yield None, f"🖼️ Generating ComfyUI Z-Image setting frame ({width}×{height})..."
        first_frame, error = None, None
        for update in generate_comfyui_zimage_first_frame(
            first_frame_prompt, f"{_safe_slug(shot_id)}_{token}", pm, width=width, height=height
        ):
            if isinstance(update, tuple):
                first_frame, error = update
            else:
                yield None, update
        if error or not first_frame:
            raise ComfyError(f"Z-Image target frame failed: {error or 'no image returned'}")

    relative = os.path.relpath(first_frame, os.path.join(pm.base_dir, pm.current_project))
    with pm.queue_lock:
        pm.df.at[row_index, "First_Frame_Image_Path"] = relative
        pm.df.at[row_index, "First_Frame_Image_Source"] = generator_name
        pm.df.at[row_index, "First_Frame_Image_Aspect"] = aspect_key
        pm.df.at[row_index, "First_Frame_Image_Prompt_Hash"] = prompt_hash
        pm.save_data()
    yield first_frame, {"prompt": first_frame_prompt}


def generate_h3_video_for_shot(shot_id, row, row_index: int, pm, source_prompt: str,
                               generation_mode="Krea 2 First Frame",
                               caching_mode="Use cached prompt",
                               use_llm_image_prompt=False):
    """Generator matching video.generate_video_for_shot's (path, status) protocol."""
    try:
        timeline_frames = int(row.get("Total_Frames", round(float(row["Duration"]) * H3_FPS)))
        render_frames = h3_render_frames(timeline_frames)
        render_duration = render_frames / H3_FPS
        settings = pm.load_project_settings()
        aspect = settings.get("h3_aspect", "3:4 - Photo")
        quality = settings.get("h3_quality", "0.65 MP - Balanced")
        custom_width = int(settings.get("h3_custom_width", 16))
        custom_height = int(settings.get("h3_custom_height", 9))
        llm_model = config.LM_STUDIO_MODEL
        job_id = f"{_safe_slug(shot_id)}_{uuid.uuid4().hex[:8]}"
        client = ComfyClient(config.H3_COMFYUI_URL)
        output_dir = Path(pm.get_path("videos"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if str(row.get("Type", "")) == "Vocal":
            lead = str(settings.get("h3_lead_character", "")).strip()
            if not lead:
                raise ComfyError("Select the H3 lead singer in Tab 2 before generating lip-sync shots.")
            face, body = h3_reference_paths(pm, lead)
            if not face or not body:
                raise ComfyError(f"Missing face/body reference images for {lead}. Generate them in Tab 2 first.")
            labels = [f"<Picture 1>: close face reference for {lead}",
                      f"<Picture 2>: full-body wardrobe reference for {lead}"]
            yield None, "🧠 Rewriting H3 lip-sync prompt..."
            prompt = rewrite_h3_prompt(pm, source_prompt, render_duration, "REF2VA", labels, llm_model)
            yield None, "🎙️ Preparing frame-accurate H3 vocal audio..."
            audio_path = _h3_audio_chunk(pm, row, render_frames, shot_id)
            yield None, "⬆️ Uploading singer references and audio to H3 ComfyUI..."
            face_name = client.upload_input(face, job_id)
            body_name = client.upload_input(body, job_id)
            audio_name = client.upload_input(audio_path, job_id)
            workflow = patch_h3_lipsync(
                face_name, body_name, audio_name, prompt, render_duration,
                f"synesthesia_h3/{job_id}", upscaled=settings.get("h3_lipsync_output", "RTX Upscaled") == "RTX Upscaled",
            )
            preferred = ["2402"] if settings.get("h3_lipsync_output", "RTX Upscaled") == "RTX Upscaled" else ["2293"]
            kind = "H3 lip-sync"
        else:
            target_frame, target_prompt = None, None
            for frame_path, update in _generate_h3_target_frame(
                shot_id, row, row_index, pm, source_prompt, generation_mode, caching_mode,
                aspect, custom_width, custom_height, use_llm_image_prompt,
            ):
                if frame_path:
                    target_frame = frame_path
                    target_prompt = update["prompt"]
                else:
                    yield None, update
            if not target_frame:
                raise ComfyError("The H3 target frame generator returned no image.")

            characters = _shot_character_names(row, pm)
            if len(characters) > H3_MAX_CHARACTERS_PER_REF2V_SHOT:
                raise ComfyError(
                    f"{shot_id} names {len(characters)} bible characters; H3 Ref2VA supports at most "
                    f"{H3_MAX_CHARACTERS_PER_REF2V_SHOT} when each needs face and body references."
                )
            if characters:
                local_paths = [target_frame]
                labels = [
                    "<Picture 1>: concrete target first frame and setting/composition anchor for this shot"
                ]
                for index, name in enumerate(characters, start=1):
                    face, body = h3_reference_paths(pm, name)
                    if not face or not body:
                        raise ComfyError(f"Missing face/body reference images for {name}. Generate them in Tab 2 first.")
                    local_paths.extend([face, body])
                    labels.extend([f"<Picture {2 * index}>: face identity reference for {name}",
                                   f"<Picture {2 * index + 1}>: full-body wardrobe reference for {name}"])
                yield None, "🧠 Rewriting H3 Ref2VA prompt..."
                prompt = rewrite_h3_prompt(pm, source_prompt, render_duration, "REF2VA", labels, llm_model)
                yield None, "⬆️ Uploading target frame and character references to H3 ComfyUI..."
                image_names = [client.upload_input(path, job_id) for path in local_paths]
                workflow = patch_h3_ref2(image_names, prompt, render_duration,
                                         f"synesthesia_h3/{job_id}", aspect, quality, custom_width, custom_height)
                preferred, kind = ["2568"], "H3 Ref2VA"
            else:
                labels = ["<Picture 1>: the supplied starting frame"]
                yield None, "🧠 Rewriting H3 FL2VA prompt..."
                prompt = rewrite_h3_prompt(pm, source_prompt, render_duration, "FL2VA", labels, llm_model)
                yield None, "⬆️ Uploading target first frame to H3 ComfyUI..."
                image_name = client.upload_input(target_frame, job_id)
                workflow = patch_h3_fl2(image_name, prompt, render_duration,
                                        f"synesthesia_h3/{job_id}", aspect, quality, custom_width, custom_height)
                preferred, kind = ["2568"], "H3 FL2VA"

        client_id = uuid.uuid4().hex
        prompt_id = client.submit(workflow, client_id)
        _record_job(pm, job_id, workflow, {"prompt_id": prompt_id, "kind": kind, "shot_id": shot_id})
        yield None, f"🎬 {kind} rendering on ComfyUI ({render_frames} frames; timeline keeps {timeline_frames})..."
        descriptor = _wait_for_output(
            client, prompt_id, preferred, {".mp4", ".webm", ".mov", ".mkv"},
            stop_check=lambda: bool(getattr(pm, "stop_video_generation", False)),
        )
        extension = Path(str(descriptor.get("filename", ""))).suffix or ".mp4"
        local_path = client.download(descriptor, output_dir / f"{_safe_slug(shot_id)}_h3_{job_id}{extension}")
        with pm.queue_lock:
            pm.df.at[row_index, "Video_Path"] = local_path
            pm.df.at[row_index, "Status"] = "Done"
            pm.df.at[row_index, "Render_Resolution"] = f"H3 {quality} {aspect}"
            pm.save_data()
        yield local_path, "Done"
    except Exception as exc:
        print(f"H3 generation failed for {shot_id}: {exc}")
        with pm.queue_lock:
            pm.df.at[row_index, "Status"] = "Error"
            pm.save_data()
        yield None, f"Error: {exc}"
