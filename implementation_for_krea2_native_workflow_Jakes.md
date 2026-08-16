# Implementing `krea2_native_workflow_Jakes version for silly hat.json`

This guide explains how to use the repository's Krea 2 ComfyUI workflow from
another Python application. It documents the integration contract that matters:
which nodes must exist, which values must be patched for each request, how to
submit the graph through ComfyUI's API, and how to retrieve the saved image.

The workflow is a **text-to-image** graph. It does not inspect the original
image itself. In Reimagine Animator, a vision model first describes the source
image and asks for the desired changes; that resulting text becomes node 6's
positive prompt. If another project needs “silly hats” or any other semantic
edit, it must put that instruction into the supplied prompt (and provide any
required LoRA). The workflow JSON does not add that instruction automatically.

## 1. What the workflow contains

The file is already in **ComfyUI API format**: a top-level object keyed by node
ID. Do not pass a UI/workflow export containing a top-level `nodes` array to
`POST /prompt`.

The execution graph is:

```text
UNET 1 ─┐
        ├─ Power LoRA 18 ─┬─ model ───────────────┐
CLIP 13 ┘                 └─ clip ─> prompt 6 ─┐  │
                                              ├─ rebalance 16 ─> positive ─┐
                                              └─ zero-out 8 ───> negative ─┤
ratio latent 15 ────────────────────────────────────────────────────────────┤
                                                                           v
                                                                      sampler 2
                                                                           │
VAE 4 ─────────────────────────────────────────────────────────────> decode 3
                                                                           │
                                                                     output 5
```

### Node contract

| ID | Class | Role | Runtime treatment |
|---:|---|---|---|
| `1` | `UNETLoader` | Loads `krea2_turbo_fp8.safetensors` | Usually left unchanged |
| `13` | `CLIPLoader` | Loads `qwen3vl_4b_fp8_scaled.safetensors` with type `krea2` | Usually left unchanged |
| `18` | `Power Lora Loader (rgthree)` | Applies `KNPV3_1.safetensors` to model and CLIP | Keep, replace, or disable deliberately |
| `6` | `CLIPTextEncode` | Positive prompt | **Replace `inputs.text` every request** |
| `16` | `ConditioningKrea2Rebalance` | Krea-specific conditioning weights | Preserve unless intentionally tuning Krea 2 |
| `8` | `ConditioningZeroOut` | Produces zeroed negative conditioning | Preserve |
| `15` | `Empty Latent by Ratio (WLSH)` | Aspect ratio, orientation, short edge, batch size | **Patch ratio/orientation/size** |
| `2` | `KSampler` | Samples the image | **Randomize `inputs.seed` every request** |
| `4` | `VAELoader` | Loads `qwen_image_vae.safetensors` | Usually left unchanged |
| `3` | `VAEDecode` | Decodes the sampled latent | Preserve |
| `5` | `PreviewImage` in the file | Terminal image output | **Convert to `SaveImage` for API retrieval** |

Node IDs are part of this integration contract. If the workflow is edited and
ComfyUI assigns new IDs, update the patcher and this table together. Checking
only that an ID exists is not enough for a long-lived integration; also verify
the expected `class_type` during startup or in tests.

## 2. Required ComfyUI installation

The JSON is not self-contained. The target ComfyUI instance must be able to
resolve all of its node classes and model filenames.

### Model files referenced by the shipped JSON

| JSON field | Required filename | Typical ComfyUI model category |
|---|---|---|
| node `1`.`unet_name` | `krea2_turbo_fp8.safetensors` | diffusion model / UNET |
| node `13`.`clip_name` | `qwen3vl_4b_fp8_scaled.safetensors` | text encoder / CLIP |
| node `4`.`vae_name` | `qwen_image_vae.safetensors` | VAE |
| node `18`.`lora_1.lora` | `KNPV3_1.safetensors` | LoRA |

Use the folders exposed by the corresponding loader dropdowns in the target
ComfyUI installation. Depending on the ComfyUI version and model-path
configuration, the diffusion model may be under `models/diffusion_models` or
`models/unet`, and the text encoder may be under `models/text_encoders` or
`models/clip`. The filename/path stored in the JSON must exactly match what the
loader node reports.

The LoRA is enabled in the shipped graph:

```json
"lora_1": {
  "on": true,
  "lora": "KNPV3_1.safetensors",
  "strength": 1
}
```

If the other project does not have or should not use this LoRA, disable/remove
that entry in ComfyUI and export a new API graph, or patch node 18 explicitly.
Do not silently rely on a missing LoRA being ignored.

### Custom nodes

The graph requires the packages that register these non-core classes:

- `Power Lora Loader (rgthree)` — from
  [rgthree-comfy](https://github.com/rgthree/rgthree-comfy).
- `Empty Latent by Ratio (WLSH)` — install the custom-node package used by the
  source ComfyUI installation, or replace node 15 with a standard latent node
  and rewrite the dimension patching.
- `ConditioningKrea2Rebalance` — install the Krea 2 node implementation used by
  the source installation. This node is part of the image-quality contract;
  deleting it changes the conditioning sent to the sampler.

The safest transfer procedure is to load the workflow in the destination
ComfyUI UI and resolve every missing/unknown node before attempting API calls.
For automated checks, query `GET /object_info/<class_type>` for every class in
the node table and fail fast when one is unavailable.

## 3. Per-request patching

Always load a **fresh JSON object for every render**. Never retain and mutate one
global dictionary: prompt, seed, dimensions, and filename prefix would leak
between requests and concurrent calls would race.

Reimagine Animator calculates a width/height pair from its UI, but the Krea 2
graph does not contain ordinary `width` and `height` inputs. Node 15 owns the
geometry through:

- `aspect`: an unoriented ratio such as `16:9` or `4:3`;
- `direction`: `landscape` or `portrait`;
- `shortside`: the desired short edge in pixels;
- `batch_size`: left at `1` in this integration.

Portrait ratios therefore map to the same base ratio with the direction
flipped:

```python
KREA2_AR_MAP = {
    "16:9": ("16:9", "landscape"),
    "9:16": ("16:9", "portrait"),
    "1:1":  ("1:1",  "landscape"),
    "4:3":  ("4:3",  "landscape"),
    "3:4":  ("4:3",  "portrait"),
    "21:9": ("21:9", "landscape"),
}
```

The current application passes `min(width, height)` as `shortside`. This keeps
the workflow's ratio node—not generic dimension math—as the final authority.

An unmapped ratio deliberately leaves node 15's file defaults (`16:9`,
`landscape`) in place while still changing `shortside`. That is safe as a
fallback but does **not** honor the requested ratio. Another project should
either extend `KREA2_AR_MAP` with a value actually accepted by its WLSH node or
reject unsupported ratios visibly.

### Reference patcher

```python
import copy
import random

KREA2_AR_MAP = {
    "16:9": ("16:9", "landscape"),
    "9:16": ("16:9", "portrait"),
    "1:1":  ("1:1", "landscape"),
    "4:3":  ("4:3", "landscape"),
    "3:4":  ("4:3", "portrait"),
    "21:9": ("21:9", "landscape"),
}

EXPECTED_NODES = {
    "2": "KSampler",
    "5": "PreviewImage",       # before conversion
    "6": "CLIPTextEncode",
    "15": "Empty Latent by Ratio (WLSH)",
}


def patch_krea2(template, prompt, width, height, filename_prefix,
                aspect_ratio="1:1"):
    # Use deepcopy only if the caller cached an immutable in-memory template.
    # json.load() per request already returns a fresh object.
    workflow = copy.deepcopy(template)

    for node_id, expected_type in EXPECTED_NODES.items():
        actual = workflow.get(node_id, {}).get("class_type")
        if actual != expected_type:
            raise ValueError(
                f"Krea 2 workflow node {node_id}: expected "
                f"{expected_type!r}, got {actual!r}"
            )

    workflow["6"]["inputs"]["text"] = prompt
    workflow["2"]["inputs"]["seed"] = random.randint(1, 10**15)

    try:
        aspect, direction = KREA2_AR_MAP[aspect_ratio]
    except KeyError as exc:
        raise ValueError(f"Unsupported Krea 2 aspect ratio: {aspect_ratio}") from exc

    workflow["15"]["inputs"].update({
        "aspect": aspect,
        "direction": direction,
        "shortside": int(min(width, height)),
        "batch_size": 1,
    })

    # PreviewImage writes a temp artifact. SaveImage creates an output artifact
    # that is stable and discoverable through /history + /view.
    workflow["5"]["class_type"] = "SaveImage"
    workflow["5"]["inputs"]["filename_prefix"] = filename_prefix

    return workflow
```

For a single-worker image queue, `reimagine_<basename>` is a sufficient staging
prefix because ComfyUI adds its own counter. If another application submits
this workflow concurrently, append a request token (for example, six UUID hex
characters) to avoid two workers choosing the same counter at save time.

## 4. Submission and result retrieval

The portable API lifecycle is:

1. Create a unique `client_id`.
2. Open `ws://<host>/ws?clientId=<client_id>`.
3. `POST /prompt` with `{"prompt": workflow, "client_id": client_id}`.
4. Save the returned `prompt_id`.
5. Read WebSocket messages until an `executing` message contains both the same
   `prompt_id` and `node: null`; that marks completion.
6. `GET /history/<prompt_id>`.
7. Inspect the history entry's `outputs` and find an image record whose
   `type` is `output`.
8. `GET /view` with that record's `filename`, `subfolder`, and `type`.
9. Save the response bytes under the destination filename your application
   owns.

Opening the WebSocket before queueing is the most robust order because an
extremely fast or cached execution cannot finish before the listener exists.
The official ComfyUI example uses the same `/prompt` → WebSocket completion →
`/history` → `/view` pattern.

### Minimal client skeleton

```python
import json
import time
import uuid
from pathlib import Path

import requests
import websocket


def run_krea2(comfy_url, workflow_path, prompt, width, height,
              aspect_ratio, destination, timeout=600):
    base = comfy_url.rstrip("/")
    with open(workflow_path, "r", encoding="utf-8") as handle:
        template = json.load(handle)

    if not isinstance(template, dict) or "nodes" in template:
        raise ValueError("Workflow must be a ComfyUI API-format object")

    token = uuid.uuid4().hex
    workflow = patch_krea2(
        template,
        prompt=prompt,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        filename_prefix=f"krea2_api/{Path(destination).stem}_{token[:6]}",
    )

    ws_base = base.replace("http://", "").replace("https://", "")
    ws_scheme = "wss" if base.startswith("https://") else "ws"
    ws = websocket.create_connection(
        f"{ws_scheme}://{ws_base}/ws?clientId={token}", timeout=30
    )
    try:
        response = requests.post(
            f"{base}/prompt",
            json={"prompt": workflow, "client_id": token},
            timeout=15,
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            ws.settimeout(min(5, max(0.1, deadline - time.time())))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not isinstance(raw, str):
                continue  # binary preview frame
            message = json.loads(raw)
            if message.get("type") != "executing":
                continue
            data = message.get("data", {})
            if data.get("prompt_id") == prompt_id and data.get("node") is None:
                break
        else:
            raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
    finally:
        ws.close()

    response = requests.get(f"{base}/history/{prompt_id}", timeout=10)
    response.raise_for_status()
    entry = response.json().get(prompt_id, {})

    candidates = []
    for node_output in entry.get("outputs", {}).values():
        if not isinstance(node_output, dict):
            continue
        for values in node_output.values():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict) and value.get("type") == "output":
                    candidates.append(value)

    if not candidates:
        raise RuntimeError(f"No saved output in history for {prompt_id}")

    image = candidates[0]
    response = requests.get(
        f"{base}/view",
        params={
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        },
        timeout=300,
    )
    response.raise_for_status()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_bytes(response.content)
    return str(destination)
```

Production code should also log ComfyUI's response body when `POST /prompt`
returns HTTP 400; its `node_errors` object usually identifies a missing node,
model, invalid enum, or malformed input directly.

## 5. Cancellation and concurrency

Closing the WebSocket does **not** cancel a ComfyUI prompt. If the host
application exposes a Stop button:

- track every `prompt_id` it submits, grouped by ComfyUI instance;
- inspect `GET /queue` before calling `POST /interrupt`;
- interrupt only when the running prompt ID belongs to the application, because
  `/interrupt` is instance-wide;
- remove owned pending IDs with `POST /queue` and
  `{"delete": ["<prompt-id>", ...]}`;
- poll `GET /queue` until the owned IDs disappear before reporting a confirmed
  stop.

Do not clear the entire ComfyUI queue unless the application exclusively owns
that server. Reimagine Animator additionally checks its stop flag after dequeue
and before forwarding work so an upstream stage cannot refill a drained queue.

Use one fresh workflow object, one seed, one client ID, and preferably one
unique filename prefix per request. If GPU memory requires serial execution,
place submissions behind a single worker or semaphore; the ComfyUI HTTP API
accepting multiple prompts does not mean the model stack can execute them safely
in parallel.

## 6. Settings worth preserving initially

The shipped sampler settings are tuned as a coherent starting point:

```text
steps       = 8
cfg         = 1
sampler     = er_sde
scheduler   = simple
denoise     = 1
rebalance   = balanced
multiplier  = 1
renormalize = true
```

Node 16 also contains the per-layer weight string:

```text
1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0
```

Transfer the workflow unchanged first. Tune sampler, conditioning, model, or
LoRA only after the API path produces a known-good image; otherwise installation
errors and quality regressions become difficult to distinguish.

## 7. Common failure modes

### `Workflow must be in API format`

The file is a normal UI export rather than an API export. In ComfyUI, export the
workflow in API format and confirm the top level is an object keyed by numeric
node IDs.

### `POST /prompt` returns 400

Read the returned `error` and `node_errors`. Typical causes are an unavailable
custom node, a model filename not present in the loader's model category, an
unsupported WLSH ratio enum, or a stale node ID/type.

### Execution completes but history has no downloadable image

Node 5 was left as `PreviewImage`, which produces a temporary result, or the
history parser assumes one hard-coded output key. Convert node 5 to `SaveImage`
and scan all list-valued output fields for records with `type: "output"`.

### Wrong orientation or size

Do not patch nonexistent `width`/`height` fields on node 15. Patch `aspect`,
`direction`, and `shortside`. Remember that `9:16` is represented as aspect
`16:9` plus direction `portrait`.

### Repeated identical images

The KSampler seed was not replaced, or the application reused a mutated cached
workflow. Reload/deep-copy the template and generate a new seed per request.

### The generated image lacks the requested semantic edit

The application did not put the edit into node 6's prompt, or the expected LoRA
was disabled/missing. “Silly hat” is not a hidden workflow operation.

### Timeouts despite an image appearing in ComfyUI

The WebSocket was connected after a very fast job completed, the client ID does
not match, or the listener accepted another prompt's completion event. Connect
before submission and require both the matching `prompt_id` and `node is None`.

## 8. Integration checklist

- [ ] Destination ComfyUI loads the workflow with no unknown nodes.
- [ ] All four referenced model/LoRA filenames resolve.
- [ ] The file is API-format JSON, not a UI graph.
- [ ] Node IDs and `class_type` values match the node contract.
- [ ] A fresh workflow object is used per request.
- [ ] Node 6 receives the final positive prompt.
- [ ] Node 2 receives a fresh seed.
- [ ] Node 15 receives a supported aspect, direction, and short edge.
- [ ] Node 5 is converted to `SaveImage` with a collision-safe prefix.
- [ ] The client correlates WebSocket messages by `prompt_id`.
- [ ] The history parser finds output records without assuming one node-output key.
- [ ] `/view` uses the returned filename, subfolder, and type.
- [ ] Stop logic cancels owned ComfyUI jobs rather than merely closing sockets.
- [ ] Errors preserve ComfyUI's `node_errors` details.

## References

- Local workflow: `krea2_native_workflow_Jakes version for silly hat.json`
- Local implementation: `KREA2_AR_MAP`, `_patch_krea2`, and
  `comfy_reimagine` in `app.py`
- [ComfyUI server routes](https://docs.comfy.org/development/comfyui-server/comms_routes)
- [Official ComfyUI WebSocket API example](https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example.py)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
