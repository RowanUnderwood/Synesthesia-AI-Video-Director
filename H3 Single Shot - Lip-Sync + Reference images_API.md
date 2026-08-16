# H3 Single Shot — API integration guide

How to drive `H3 Single Shot - Lip-Sync + Reference images_API.json` from another
application.

One POST produces **one lip-synced shot of up to 15 seconds** from two character
reference images, an audio clip, and a prompt. It writes two MP4s: native resolution and
an RTX VSR 2× upscale. There is no shot sequencing, no checkpointing, and no continuity
state — the calling application owns all of that.

---

## Contents

- [What it produces](#what-it-produces)
- [Prerequisites](#prerequisites)
- [The five inputs you change per shot](#the-five-inputs-you-change-per-shot)
- [End-to-end client](#end-to-end-client)
- [Hard rules](#hard-rules) ← **read this before writing any code**
- [Node map](#node-map)
- [Validating a prompt before you ship it](#validating-a-prompt-before-you-ship-it)
- [Measured performance](#measured-performance)

---

## What it produces

For each queued prompt, in `ComfyUI/output/h3_shot/`:

| File | From node | Resolution | Frames | Duration |
| --- | --- | --- | --- | --- |
| `shot_NNNNN-audio.mp4` | `2293` | 960 × 544 | 362 | 15.083 s |
| `shot_rtx_NNNNN-audio.mp4` | `2402` | 1920 × 1088 | 362 | 15.083 s |

Both carry a 32 kHz stereo AAC track. `NNNNN` is a counter ComfyUI assigns; never
predict it — read it back from the history response.

VHS also writes a video-only `shot_NNNNN.mp4` and a workflow-metadata `shot_NNNNN.png`
alongside each. **The file your application wants is the `-audio.mp4` one.** The bare
`.mp4` is the pre-mux intermediate and has no sound.

---

## Prerequisites

Verified against this exact stack:

| Component | Version |
| --- | --- |
| ComfyUI | 0.33.0 |
| ComfyUI frontend | 1.49.6 |
| Python | 3.10.11 |
| `ComfyUI-VideoHelperSuite` | 1.7.7 |
| `ComfyUI-DaSiWa-Nodes` | 0.4.4 |
| `ComfyMath` | 0.1.0 |
| `ComfyUI-H3-Motion-Context-MultiRef` | (unversioned) |

`MiniMaxH3ReferenceToVideo`, `ResolutionSelector`, `SamplerCustomAdvanced` and the
primitives are **ComfyUI core** — no pack needed.

Models, all of which must resolve on the target server:

```
unet   minimax_h3_fl2va_pruned_int8_convrot.safetensors
clip   qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae    minimax_h3_video_vae_int8_convrot.safetensors      (video)
vae    minimax_h3_audio_vae_fp32.safetensors              (audio)
lora   minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

The RTX upscale additionally needs the `nvvfx` Python package (`nvidia-vfx`) in
ComfyUI's venv. It bundles its own runtime — `NVVideoEffects.dll`, `nvngx_vsr.dll`,
TensorRT 10 — so **no separate NVIDIA Broadcast/Video SDK install is required**, despite
what the node's error message claims. It needs a Turing-or-newer GPU and driver 570.65+.

If you do not want the upscale, delete nodes `2401` and `2402` from the prompt. Nothing
else references them.

---

## The five inputs you change per shot

Everything else in the JSON is fixed configuration. Load the file, patch these, POST.

| Node | Field | Meaning |
| --- | --- | --- |
| `910` | `inputs.image` | Face / identity reference — filename in ComfyUI's `input/` |
| `911` | `inputs.image` | Full-body / wardrobe reference — same performer |
| `940` | `inputs.audio` | This shot's audio clip — **must be ≥ 15.084 s**, see [Rule 3](#rule-3--the-audio-clip-must-be-at-least-length--24-seconds) |
| `110` | `inputs.prompt` | The H3 REF-format prompt (six sections) |
| `120` | `inputs.noise_seed` | Change to regenerate; keep to reproduce |

Optional:

| Node | Field | Default | Notes |
| --- | --- | --- | --- |
| `101` | `inputs.value` | `15` | Shot duration in seconds. Clamped to 362 frames — see [Rule 4](#rule-4--length-is-a-17k5-frame-grid-capped-at-362) |
| `100` | `inputs.megapixels` | `0.5` | `0.5` → 960 × 544. Raising this raises VRAM sharply |
| `970` | `inputs.value` | `8` | Sampler steps; the Turbo LoRA is trained for 8 |
| `2293` / `2402` | `inputs.filename_prefix` | `h3_shot/shot`, `h3_shot/shot_rtx` | Per-job output paths |

ComfyUI caches by input signature, so changing any of these correctly invalidates the
downstream nodes and re-runs only what is affected.

---

## End-to-end client

```python
import json, time, urllib.request, urllib.parse, uuid, mimetypes

HOST = "http://127.0.0.1:8188"

def upload(path, overwrite=True):
    """Put a local file into ComfyUI's input/ dir. Returns the name to reference.

    The form field is literally called "image" for ALL file types, audio included --
    the handler writes the bytes verbatim with no type check.
    """
    name = path.split("/")[-1]
    boundary = "----" + uuid.uuid4().hex
    with open(path, "rb") as fh:
        payload = fh.read()
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="{name}"\r\nContent-Type: '
        f'{mimetypes.guess_type(name)[0] or "application/octet-stream"}\r\n\r\n'.encode(),
        payload, b"\r\n",
        f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\n'
        f'{"true" if overwrite else "false"}\r\n'.encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        HOST + "/upload/image", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    # ALWAYS use the returned name -- see Rule 5
    return json.load(urllib.request.urlopen(req))["name"]


def render_shot(workflow_path, face, body, audio, prompt_text, seed, duration=15.0):
    with open(workflow_path, encoding="utf-8") as fh:
        wf = json.load(fh)

    wf["910"]["inputs"]["image"] = upload(face)
    wf["911"]["inputs"]["image"] = upload(body)
    wf["940"]["inputs"]["audio"] = upload(audio)
    wf["110"]["inputs"]["prompt"] = prompt_text
    wf["120"]["inputs"]["noise_seed"] = seed
    wf["101"]["inputs"]["value"] = duration

    body_ = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=body_,
                                 headers={"Content-Type": "application/json"})
    try:
        queued = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as exc:
        # 400 means validation failed; node_errors names the node and the input
        raise RuntimeError(exc.read().decode()) from None

    pid = queued["prompt_id"]
    while True:
        hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
        if pid in hist:
            break
        time.sleep(2)

    entry = hist[pid]
    if entry["status"]["status_str"] != "success":
        raise RuntimeError(f"render failed: {entry['status']}")

    out = entry["outputs"]
    return {
        "native": out["2293"]["gifs"][0],   # 960x544
        "upscaled": out["2402"]["gifs"][0],  # 1920x1088
    }
```

Each returned entry is `{"filename", "subfolder", "type", "format", "frame_rate",
"fullpath"}`. Fetch the bytes over HTTP with:

```
GET /view?filename=<filename>&subfolder=<subfolder>&type=output
```

or read `fullpath` directly if your application shares a filesystem with ComfyUI.

---

## Hard rules

Every one of these was a real failure during integration. Four of them fail **silently**.

### Rule 1 — Dynamic inputs keep their dotted names

`ComfyMathExpression` and `MiniMaxH3ReferenceToVideo` use ComfyUI's
`COMFY_AUTOGROW_V3` inputs. In an API prompt these keep their **dotted** names:

```jsonc
"102": { "inputs": { "values.a": ["101", 0] } }
"110": { "inputs": { "ref_images.ref_image_0": ["910", 0],
                     "ref_images.ref_image_1": ["911", 0] } }
```

Not `a`. Not `ref_image_0`. ComfyUI nests them at execution time via
`_io.build_nested_inputs()`.

> **This one is a trap.** A wrong name on a *required* dynamic input fails loudly with
> `required_input_missing`. A wrong name on an *optional* one — which is what
> `ref_images` is — is **silently ignored**: the shot renders with **no reference
> images at all**, no warning anywhere, and a perfectly valid-looking MP4 of the wrong
> person.

`GET /object_info` will not help you here. It reports these inputs *collapsed* under
their group name (`ref_images`, `values`) because V3 nodes serve that endpoint through a
different code path than the one validation uses. Trust the JSON as shipped.

### Rule 2 — `trim_to_audio` must stay `false`

VideoHelperSuite reads this widget **inverted** from what its name suggests. `true`
means "trim the *video* to the audio": it drops the `apad` filter while keeping
`-shortest`. H3's decoded audio lands a few frames shorter than the picture, so the
render gets cut.

Measured on a 362-frame shot: `true` produced **359** frames, `false` produces all
**362**. The cost of `false` is roughly 0.1 s of silence padded onto the tail, which is
the right way round.

Both `2293` and `2402` ship with `false`. Leave them.

### Rule 3 — The audio clip must be at least `length / 24` seconds

For the default 362 frames that is **15.0833 s** — not 15.0.

`MiniMaxH3SongMaskedAVContext` slices exactly `length / 24` seconds from the file. A
short file is tail-padded with silence and only writes a log line:

```
master audio slice is N samples short
```

Your application will never see it. An audio pipeline that cuts shots to a round 15.000 s
silently loses lip-sync on the last two frames of every shot. Cut long — a second of
slack costs nothing, since only the needed span is encoded.

### Rule 4 — `length` is a 17k+5 frame grid, capped at 362

Valid H3 video runs are `5, 22, 39, … 17k + 5`. Node `102` converts seconds to the
nearest valid count and clamps it:

```
min(max(5, round(a * 24)), 362) + (5 - (min(max(5, round(a * 24)), 362) % 17)) % 17
```

- `15 s` → **362 frames** → 15.0833 s (`362 = 5 + 17 × 21`)
- `5 s` → 124 frames
- anything above 15 s clamps to 362

362 is the top of the model's documented trained range (`length` tooltip: *"trained
range is ~124-362"*), and it is what every bundled H3 example workflow uses. The widget
itself accepts up to 3600, but past 362 you are off the trained range.

Set `101.value` in **seconds** and let node `102` do the conversion. Do not write
`110.inputs.length` directly — it is a link, and a literal there would be overwritten.

### Rule 5 — Uploads auto-rename on collision

`/upload/image` compares hashes. If a *different* file with the same name already exists,
it saves as `name (1).ext` and returns that in `resp["name"]`.

A per-shot pipeline that always uploads `shot.wav` and then references `"shot.wav"` will
render **the first shot's audio forever**. Either pass `overwrite=true`, or use the
returned name. The client above does both.

### Rule 6 — Leave `ref_audio` unconnected

`MiniMaxH3ReferenceToVideo` has `ref_audios.ref_audio_N` inputs. **Do not wire them.**

H3 treats reference audio as a *guide* and re-generates its own similar-sounding
approximation instead of matching the supplied timing — lip-sync drifts. This workflow
instead encodes the exact audio interval into the H3 audio latent and protects it with a
per-token noise mask of 0 (`MiniMaxH3SongMaskedAVContext`), so there is nothing to
regenerate and nothing can drift.

The prompt text shipped in node `110` still mentions `<Audio 1>` in three places. That is
stale copy from an earlier architecture, inert because no reference audio is wired. If
you author new prompts, drop those references.

### Rule 7 — `device_id` is `0`, whatever `nvidia-smi` says

ComfyUI launches with `--cuda-device N`, so torch sees exactly **one** GPU and it is
always `cuda:0`. The RTX node's `device_id` must therefore be `0`. Setting anything else
fails.

CUDA orders devices `FASTEST_FIRST` by default, which is **not** the PCI-bus order
`nvidia-smi` prints — `--cuda-device 1` on the reference machine selects an RTX 5090
that `nvidia-smi` lists at index 0. To find out which GPU is actually in use, read
`GET /system_stats`, never nvidia-smi indices.

### Rule 8 — Some VHS inputs are undeclared, and required anyway

`pix_fmt`, `crf`, `save_metadata` and `trim_to_audio` on `VHS_VideoCombine` are not
declared inputs — they are widgets belonging to the `video/h264-mp4` **format**, which
`combine_video(**kwargs)` forwards to `apply_format_widgets`. ComfyUI ignores undeclared
prompt keys during validation, so they pass through fine.

Do not "clean them up" because a schema check flags them. Removing them makes VHS
substitute its own defaults — including `trim_to_audio = false`'s opposite — and you are
back to [Rule 2](#rule-2--trim_to_audio-must-stay-false).

### Rule 9 — The RTX node has a fixed 8 GiB GPU-output ceiling

`DaSiWa_RTX_UpscalerRefiner` pre-allocates its whole output tensor. For a **CPU** input
it can fall back to a disk-backed mmap (up to 64 GiB); for a **GPU** input it raises
above a hard 8 GiB, which is a node constant, not a VRAM limit.

At 362 × 1920 × 1088 fp32 the output is **8.86 GiB**. ComfyUI's `VAEDecode` returns on
the intermediate device (CPU) by default, so this is fine — but launching ComfyUI with
`--gpu-only` breaks it. If you must, drop `2401.inputs.scale` to `1.5`.

Scaling roughly: output bytes ≈ `frames × width × height × 3 × 4 × scale²`.

---

## Node map

```
UNETLoader(1) → AttentionBackend(1756) → TurboLoRA(977) → SigmaShift(5) ─┬─→ BasicGuider(121) ─┐
                                                                        └─→ Scheduler(123) ───┤
CLIPLoader(2) ─┐                                                          Steps(970) ─────────┘│
VAELoader(3,4) ├─→ MiniMaxH3ReferenceToVideo(110) ─┬─ positive ────────────────────────────────┘
LoadImage(910,911) ┘   960×544, length ← 362       └─ LATENT ─→ SongMaskedAVContext(1755) ──┐
ResolutionSelector(100) ┘                             LoadAudio(940) ─────────┘             │
Duration(101) → FrameLength(102) ┘                                                          │
                                                                                            ▼
                                              RandomNoise(120), KSamplerSelect(973) → SamplerCustomAdvanced(124)
                                                                                            │
                                            ┌───────────────────────────────────────────────┤
                                            ▼                                               ▼
                                     VAEDecode(2291)                             VAEDecodeAudio(2292)
                                            │                                               │
                          ┌─────────────────┴──────────────┐                                │
                          ▼                                ▼                                │
                 VHS_VideoCombine(2293)        DaSiWa_RTX_Upscaler(2401)                     │
                   h3_shot/shot                          │                                   │
                          ▲                              ▼                                   │
                          └──── audio ──────── VHS_VideoCombine(2402) ◄── audio ──────────────┘
                                                 h3_shot/shot_rtx
```

Why the audio latent is locked and the video is free: with `vae` and `source_frames`
unconnected and `context_length = 0`, `MiniMaxH3SongMaskedAVContext` leaves `video_mask`
all `1.0` (every frame free to denoise — this is a fresh shot, not a continuation) and
sets `audio_mask` all `0.0` (the encoded audio never denoises).

---

## Validating a prompt before you ship it

Two scripts sit next to the workflow:

```bash
# offline: structure, reachability, shot config, API↔graph parity
python verify_single_shot.py

# live: class_types, input names, and that every model/image/audio actually exists
python check_api_against_comfy.py [http://127.0.0.1:8188]
```

`check_api_against_comfy.py` is the one to run in CI against a real server — it catches a
renamed checkpoint, a missing LoRA, or an absent input file before a job burns GPU time.
It expands autogrow templates the way validation does, so it agrees with
[Rule 1](#rule-1--dynamic-inputs-keep-their-dotted-names) rather than with
`/object_info`.

It prints eight expected notes about the undeclared `VHS_VideoCombine` widgets from
[Rule 8](#rule-8--some-vhs-inputs-are-undeclared-and-required-anyway). Those are
informational; exit code 0 is the thing to check.

Neither script substitutes for a real POST. ComfyUI returns HTTP 400 with a `node_errors`
object naming the offending node and input — surface that verbatim in your logs.

---

## Measured performance

RTX 5090, 362 frames at 960 × 544, 8 steps `res_multistep`:

| | |
| --- | --- |
| Cold run, everything executed | **150 s** |
| Re-run with only output settings changed | **10 s** (sampler, decode and upscale all cached) |
| Peak VRAM | ~27 GB |

The RTX pass runs one frame at a time with a CUDA sync either side, so it does not
pipeline; enabling `denoise` or `deblur` adds roughly another full pass each over all 362
frames. Both ship `false` — H3 output is clean diffusion output rather than compressed
camera footage, so they mostly cost time.

If you queue shots back to back, keep the model-loading nodes byte-identical between
prompts. ComfyUI will then cache the whole model stack and only the sampler onward
re-runs.
