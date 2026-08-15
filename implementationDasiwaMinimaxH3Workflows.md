# Integrating the MiniMax H3 FL2VA API workflow

This guide explains how to use
`DasiwaMinimaxH3WorkflowsT2VA_cMMH3V10_jakes version_API.json` in another
Python application that submits jobs to ComfyUI. It covers the integration
boundary: prepare a reference image, patch a fresh API workflow, submit it,
and retrieve the generated video.

The workflow is an **API-format ComfyUI prompt**. It is not a UI workflow and
should be sent to `POST /prompt` as the value of the `prompt` field.

## Prerequisites

The target ComfyUI installation must have the checkpoints, custom nodes, and
models required by the exported workflow. In particular, the workflow relies
on these custom node types:

- `MiniMaxH3Director`
- `DaSiWa_ResolutionScaleCalculator`
- `DaSiWa_RTX_UpscalerRefiner`
- `DaSiWa_EnhancedVideoCombine`
- `DaSiWa_LTX2LoraLoader`
- `MiniMaxH3SigmaShift`

Confirm the instance is reachable before submitting work (for example, with
`GET /system_stats`). The application in this project uses the video ComfyUI
endpoint `http://127.0.0.1:8189` by default, but the URL should be a setting
in a reusable integration.

Keep the supplied workflow file immutable. Load it once, validate it, and
deep-copy it for every job. This prevents one request's prompt, seed, image,
or output prefix from leaking into another request.

## Workflow nodes this integration patches

The supplied workflow uses stable node IDs. Treat them as part of this
workflow version's contract; if you replace the exported JSON, re-inspect and
revalidate them.

| Node ID | Node role | Inputs normally patched |
| --- | --- | --- |
| `2693` | MiniMax H3 Director | `mode`, `duration`, `prompt`, `timeline_data` |
| `1512:2531` | Resolution calculator | custom aspect dimensions and aspect enum |
| `1512:2528` | RTX upscaler/refiner | output `width`, `height`, ratio, resize mode |
| `1512:2600` | Random noise | `noise_seed` |
| `2568` | Video combine/output | `filename_prefix`, `save_output` |
| `2678` | Optional LoRA loader | JSON-string `stack_data` and `model_type` |
| `1512:2590`, `1512:2679` | Video/audio schedulers | Turbo step counts, when used |
| `1512:2692`, `1512:2691` | Sigma shifts | Turbo shifts, when used |

## Upload the reference image first

Do not put a local path such as `C:\\images\\reference.png` in the workflow.
ComfyUI needs an image in its input storage. Upload the exact normalized image
or extracted video frame that will be used for generation:

```python
with open(reference_path, "rb") as image:
    response = client.post(
        f"{base_url}/upload/image",
        files={"image": (Path(reference_path).name, image)},
        data={"type": "input", "overwrite": "true"},
        timeout=30,
    )
response.raise_for_status()
uploaded = response.json()

# Preserve the returned subfolder when one is supplied.
image_name = "/".join(
    part for part in (str(uploaded.get("subfolder") or "").strip("/\\"),
                       str(uploaded["name"])) if part
)
```

Use a job-unique upload filename when requests can overlap or when multiple
ComfyUI instances share the same `input` directory. This avoids one job
overwriting another job's reference.

## Patch the Director timeline correctly

`2693.inputs.timeline_data` is a **string containing JSON**, not an outer JSON
object. Build and serialize it separately. The Director receives:

- `mode`: exactly `FL2VA`.
- `duration`: the requested generated clip length (for example, `5`).
- `prompt`: an empty string.
- `timeline_data`: the image and prompt data shown below, serialized with
  `json.dumps`.

The prompt must be mirrored in both `items[0].prompt` and
`prompt_blocks[0].text`. Keep both timeline `duration` values at `1`; those
are timeline-slot units, not the generated clip duration.

```python
item_id = f"image-{uuid.uuid4().hex}"
timeline = {
    "version": 1,
    "items": [{
        "id": item_id,
        "enabled": True,
        "order": 0,
        "slot": 0,
        "start": 0,
        "duration": 1,
        "type": "image",
        "value": image_name,       # filename returned by /upload/image
        "prompt": prompt,
    }],
    "prompt_blocks": [{
        "id": f"attached-{item_id}",
        "text": prompt,
        "enabled": True,
        "start": 0,
        "duration": 1,
        "order": 0,
    }],
}

director = workflow["2693"]["inputs"]
director["mode"] = "FL2VA"
director["duration"] = int(duration)
director["prompt"] = ""
director["timeline_data"] = json.dumps(
    timeline, ensure_ascii=False, separators=(",", ":")
)
```

## Patch resolution, seed, and output

Set the seed at `1512:2600.inputs.noise_seed`. Set the output dimensions at
`1512:2528.inputs.width` and `.height`, use `resize_type = "Dimensions"`, and
set `ratio_preset` when your application exposes preset aspect ratios.

For a selected source or custom aspect ratio, patch the resolution calculator:

```python
resolution = workflow["1512:2531"]["inputs"]
resolution["scale_from_image"] = False
resolution["custom_aspect_width"] = width
resolution["custom_aspect_height"] = height
resolution["aspect_preset_when_not_image"] = "CUSTOM"
workflow["1512:2600"]["inputs"]["noise_seed"] = int(seed)

combine = workflow["2568"]["inputs"]
combine["filename_prefix"] = unique_output_prefix
combine["save_output"] = True
```

`CUSTOM` is case-sensitive. `Custom` is invalid for this node. Use a unique
output prefix per job, especially for concurrent jobs.

## Minimal submission flow

```python
template = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
workflow = copy.deepcopy(template)

# Upload reference, then apply the Director and other patches described above.
payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}
response = client.post(f"{base_url}/prompt", json=payload, timeout=30)
response.raise_for_status()
submitted = response.json()
if submitted.get("error"):
    raise RuntimeError(submitted.get("node_errors") or submitted["error"])
prompt_id = submitted["prompt_id"]
```

Monitor `ws://<host>/ws?clientId=<client_id>` (or `wss` for HTTPS) for progress
events, and poll `GET /history/{prompt_id}` as the authoritative completion
record. If the history status is `error`, surface its messages to the user.
Cancellation normally removes the queued prompt with `POST /queue` using
`{"delete": [prompt_id]}` and may call `POST /interrupt` for currently running
work.

## Download the actual output type

Search `history[prompt_id]["outputs"]` recursively for a video descriptor
(`filename`, `subfolder`, and `type`). Download it from `GET /view` using those
values as query parameters. Do not assume MP4: this workflow can produce
AV1/Opus WebM depending on the ComfyUI output settings. Preserve the extension
from ComfyUI's returned filename, then transcode to H.264/AAC MP4 only if your
player or downstream pipeline requires it. Stream the response to disk for
large outputs rather than buffering it in memory.

## Optional MiniMax Turbo LoRA mode

Turbo settings are an application feature, not a requirement for normal
operation. Standard mode preserves the workflow's sampling defaults (25 steps
and its exported sigma shifts). Before enabling a Turbo profile, verify the
LoRA exists on every target ComfyUI instance, then patch only a managed or
empty slot in `2678.inputs.stack_data` (another JSON string). Do not overwrite
unrelated character/style LoRAs. Patch both scheduler nodes and both sigma
nodes consistently with the chosen profile.

## Guardrails and regression tests

Validate the expected node IDs before accepting a template. At a minimum, add
tests that assert each job receives an independent deep copy and that:

- the uploaded ComfyUI filename is in `timeline_data.items[0].value`;
- Director mode is `FL2VA`, its direct prompt is blank, and its clip duration
  is the requested value;
- both nested prompt fields match;
- both nested timeline durations remain `1`;
- custom aspect uses uppercase `CUSTOM`;
- a WebM response is saved with `.webm`, rather than WebM bytes being named
  `.mp4`.

Persist each submitted workflow in an application-managed job directory while
debugging. It makes node-validation errors and workflow-version changes much
easier to diagnose without ever altering the canonical template.
