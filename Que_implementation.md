# Reimagine Animator Queue Implementation

## Purpose

This document explains how Reimagine Animator overlaps prompt generation, image
generation, and video generation so that the GPUs and LM Studio spend less time
idle. It is intended as a practical guide for porting the same speedups into
another video-generation project.

The requested filename uses **Que**, but the conventional technical spelling is
**queue**, which is used throughout this guide.

The important idea is not merely to put all shots in one list. Reimagine
Animator uses a staged asynchronous pipeline, independent worker pools, a shared
LLM concurrency limit, and an optional second video worker that borrows the 4090
after image work has drained.

## Architecture at a glance

```text
original image
      |
      v
q_v: Vision Prompt
4 worker tasks ----\
                   +-- shared LM Studio semaphore: maximum 4 chats total
      |            |
      v            |
q_c: Image Generation
1 worker on 4090
      |
      v
q_vp: Video Prompt
4 worker tasks ----/
      |
      v
q_l: Video Generation
1 primary worker on 5090
      |
      +-- optional co-op worker on 4090 after q_v, q_c, and q_vp drain
```

For the normal **Render Both** path, each version of each shot moves through
four stages:

1. The 3090/LM Studio analyzes the original and creates an image prompt.
2. the 4090/ComfyUI generates the reimagined image.
3. The 3090/LM Studio analyzes that image and creates a video prompt.
4. The 5090 generates the video. With MiniMax co-op enabled, the 4090 later
   joins this stage as a second video renderer.

These stages overlap. The 5090 can begin rendering the first completed video
job while the 4090 is still creating later images and LM Studio is still making
later prompts. Once all image and prompt work is complete, the otherwise-idle
4090 is reassigned to video generation.

## Why this is faster

A sequential implementation processes one shot from beginning to end before it
starts the next shot. During an LLM call, both render GPUs may be idle. During a
long video render, the image GPU and LLM may be idle.

The staged queue keeps each resource fed independently:

- Up to four LLM conversations run concurrently.
- Image generation starts as soon as any image prompt is ready.
- Video-prompt generation starts as soon as any reimagined image is ready.
- The 5090 starts video generation as soon as any video prompt is ready; it does
  not wait for the whole image batch.
- After upstream work drains, the 4090 consumes the same video queue alongside
  the 5090.
- Cached assets can enter at a later queue instead of repeating earlier stages.

The queues also provide back-pressure naturally. A slow video renderer can
accumulate video jobs without blocking prompt or image workers, and each GPU
stage remains serialized unless the system intentionally adds another physical
renderer.

## Core data model

Reimagine Animator creates one `asyncio.Queue` for each stage:

```python
queues = {
    "v":  asyncio.Queue(),  # original -> image prompt
    "c":  asyncio.Queue(),  # image prompt -> reimagined image
    "vp": asyncio.Queue(),  # reimagined image -> video prompt
    "l":  asyncio.Queue(),  # video prompt -> video file
}
```

The current queue-item shapes are:

```text
vision:  (filename, project, version_tag, forward)
comfy:   (filename, project, image_prompt, version_tag, forward)
vprompt: (filename, project, image_prompt, reimagined_path, version_tag)
video:   (filename, project, reimagined_path, video_prompt, version_tag)
```

`forward=False` makes an image job terminal at the image stage for an
images-only run. In a new project, a typed job object is preferable to tuples:

```python
@dataclass
class GenerationJob:
    job_id: str
    project_id: str
    source_path: Path
    shot_name: str
    version: str
    terminal_stage: str = "video"
    image_prompt: str | None = None
    image_path: Path | None = None
    video_prompt: str | None = None
```

A stable `job_id` is useful for cancellation, retry tracking, deduplication,
logs, and UI updates. `terminal_stage` also makes generalized ETA accounting
more accurate for image-only or otherwise short-circuited jobs.

## Worker topology

Each run starts this worker set:

| Stage | Worker tasks | Physical resource |
|---|---:|---|
| Vision/image prompt | 4 | LM Studio on 3090 |
| Image generation | 1 | image ComfyUI on 4090 |
| Video prompt | 4 | LM Studio on 3090 |
| Video generation | 1 initially | video backend on 5090 |
| Video co-op | 0 or 1 dynamically | image ComfyUI on 4090 |

There are four vision workers and four video-prompt workers, but this does **not**
mean eight simultaneous LLM requests. Both pools share one semaphore:

```python
llm_slots = asyncio.Semaphore(4)

async with llm_slots:
    result = await loop.run_in_executor(None, blocking_llm_call, inputs)
```

The shared semaphore caps LM Studio at **four concurrent conversations across
both prompt stages combined**. This matters once the pipeline is full: image
prompts and video prompts may be ready at the same time. Separate four-slot
semaphores would accidentally permit eight calls and could cause memory
pressure, model-server failures, or worse per-request throughput.

The synchronous HTTP/generation functions run in an executor (or via
`asyncio.to_thread`) so they do not block the event loop. The event loop must
remain free to move jobs, observe cancellation, update status, and supervise the
4090 co-op.

## The worker contract

Every stage follows the same lifecycle:

1. Await a queue item.
2. Recognize a `None` sentinel as shutdown.
3. If a hard stop is active, discard the item and fix its pending count.
4. Decrement `pending` and mark the job active.
5. Acquire the resource semaphore if the resource has one.
6. Execute the blocking service call outside the event-loop thread.
7. On success, increment `done`.
8. If the job continues, put the transformed job into the next queue and
   increment that stage's `pending` count.
9. Clear active state and call `task_done()` in `finally`.

Portable pseudocode:

```python
async def stage_worker(in_q, out_q, stage, next_stage, transform, slots=None):
    while True:
        job = await in_q.get()
        try:
            if job is None:
                return
            if hard_stop.is_set():
                status.add_pending(stage, -1)
                continue

            status.claim(stage, job.job_id)  # pending -1, active +job
            if slots:
                async with slots:
                    result = await asyncio.to_thread(transform, job)
            else:
                result = await asyncio.to_thread(transform, job)

            if result is not None and not hard_stop.is_set():
                status.complete(stage)       # done +1, record timestamp
                if out_q is not None and job.terminal_stage != stage:
                    await out_q.put(result)
                    status.add_pending(next_stage, 1)
        except Exception as exc:
            status.fail(stage, job.job_id, exc)
        finally:
            if job is not None:
                status.release(stage, job.job_id)
            in_q.task_done()
```

The stop check after the external call is load-bearing. A call may finish just
after the user presses Stop; its output must not be forwarded into a queue that
the stop routine has already drained.

## Job routing and cache-aware entry points

Not every request needs to start at vision. Reimagine Animator's shared job
builder routes work to the earliest necessary stage:

| Request/cache state | Entry queue |
|---|---|
| Regenerate everything | Vision |
| Reuse image prompt | Image generation |
| Reuse reimagined image | Video prompt |
| Reuse image and video prompts | Video generation |
| Video only, no cached video prompt | Video prompt |
| Video only with cached video prompt | Video generation |
| Images only | Vision or image generation |

The initial run and mid-run injection use the same job builder. This avoids a
common porting bug where a single-shot regenerate button handles cache modes
differently from a batch run.

When injecting work from a UI thread, mutate `asyncio.Queue` on its owning event
loop:

```python
pipeline_loop.call_soon_threadsafe(inject_jobs_on_loop)
```

Update the corresponding `pending` counters and run total at the same logical
time. New upstream work also tells the co-op supervisor to return the 4090 to
image generation.

## 4090 + 5090 video co-op

### Intended behavior

The 5090 is the primary video GPU and can render videos while later shots still
move through the upstream stages. The 4090 remains the exclusive image GPU
during that period.

When **all** image and prompt work is finished, the 4090 changes roles:

```text
Before upstream drain:
4090 = image generation
5090 = video generation

After upstream drain:
4090 = video generation worker #2
5090 = video generation worker #1
```

Both workers consume the same FIFO video queue. `asyncio.Queue` performs the
load distribution automatically: whichever GPU worker becomes available first
claims the next clip. This is preferable to pre-splitting the batch because the
two GPUs may have different render times and individual clips may vary in cost.

### Safe engagement gate

Reimagine Animator starts the co-op only when all of these are true:

- The feature is enabled.
- The selected backend is ComfyUI MiniMax H3; LTX has no second compatible
  backend instance to borrow.
- Power mode is not **Alternate between 4090 and 5090**.
- The run is not images-only.
- Vision, image generation, and video-prompt stages have no pending or active
  work.
- Their actual queues are also empty.
- At least two clips remain queued for video (`COOP_MIN_QUEUE = 2`).
- A previously retired co-op worker has completely exited.

Checking only the image queue is insufficient. The final generated image still
needs an LLM video-prompt call, and mid-run injections can populate an upstream
queue between supervisor polls. Check both status and the real queues.

The two-clip threshold avoids paying model-unload and model-load costs when
there is no second queued clip for the 4090 to claim.

### Memory preparation

Before starting MiniMax on the 4090, the supervisor:

1. Calls ComfyUI `/free` on the image instance with model unloading and memory
   release enabled.
2. Unloads LM Studio's resident models.
3. Starts a second video worker pointed at the image ComfyUI instance.

This project does that because two MiniMax instances have high Windows commit
charge. Measured locally, one instance reserved about 77 GB of commit and the
LLM about 19.5 GB. A port must size and monitor the **system commit limit**, not
only GPU VRAM or process working set. A production port should gate co-op
engagement on measured available commit and retain a safety margin.

LM Studio is expected to JIT-load again when a later prompt request arrives.
Consequently, the LLM client must tolerate the model-load race and retry
transient 5xx/not-loaded responses with stop-aware backoff.

### Co-op retirement

If new vision, image, or video-prompt work is injected, the supervisor sets a
private stop event for the co-op worker. The worker finishes its current clip
and exits before claiming another. This safely returns the 4090 to image work.

Do not retire the co-op with a shared queue sentinel:

- A sentinel is appended to the back of a deep FIFO queue, so it is not prompt.
- Either video worker could consume it, accidentally stopping the 5090 instead
  of the 4090.

The co-op worker therefore polls `q_l.get()` with a short timeout and checks its
private event between claims. The primary worker continues using a normal
blocking `get()`.

Do not re-engage until the retired worker's task reports `done()`. Otherwise the
system can briefly run the primary plus the old co-op plus a new co-op—three
video renders instead of two.

### Multi-worker status and output safety

A single `active_filename` value cannot represent two simultaneous video
renders. Reimagine Animator keeps a set of active video basenames and derives
the UI string from that set. Completion detection uses the set-backed status so
one worker finishing cannot make the pipeline appear idle while the other is
still rendering.

Both ComfyUI instances share an output directory. Each video save receives a
random token in its `filename_prefix`; otherwise two simultaneous saves could
inspect the directory, choose the same numeric suffix, and overwrite each
other. A port should use globally unique staging names or separate output
directories.

## Run completion and worker shutdown

Workers remain alive after the initial queues drain so a user can inject
single-shot regenerations into the active run. Natural completion requires:

- every queue to be empty; and
- every stage to have zero pending work and no active jobs.

Reimagine Animator requires this idle condition for three consecutive 0.5-second
checks. The debounce prevents a transient handoff between stages from being
mistaken for completion.

After completion, `None` sentinels are placed into each queue—one per worker
that might consume that queue—and all tasks are gathered. The dynamic video
worker count is important because co-op can add a worker during the run.
Reimagine Animator increments the video worker count when co-op starts and does
not decrement it when co-op retires. An extra sentinel left on a drained queue
is harmless; one missing sentinel can leave teardown waiting forever.

## Hard Stop behavior

Stop means cancel all pipeline-owned generation, including active image work on
the 4090 and active video work on both the 4090 and 5090.

The stop sequence is:

1. Set a thread-safe hard-stop event immediately.
2. Enter a distinct `stopping` phase; do not report completion yet.
3. Schedule queue draining on the pipeline event-loop thread.
4. Remove unclaimed jobs from all four queues and decrement pending counts.
5. Have every worker check the stop event after dequeue and before forwarding.
6. Cancel pipeline-owned prompt IDs in both ComfyUI instances.
7. Interrupt ComfyUI and call the LTX cancel endpoint when LTX is active.
8. Add the correct number of sentinels and await every worker.
9. Run a second backend-cancellation pass to catch a prompt ID returned during
   the first cancellation snapshot.
10. Only then publish `stopped`.

Track backend job IDs owned by this pipeline. Never clear an entire shared
ComfyUI queue indiscriminately, because it may contain work submitted by another
application.

A separate `stopping` phase is necessary because the simple `running` flag is
cleared as soon as Stop is pressed. Starting a new run or deleting a project
must be blocked while either `running` is true **or** phase is `stopping`; active
backend calls may still be unwinding during that window.

## ETA system

### What is measured

The ETA estimator records a monotonic timestamp whenever a stage increments its
`done` count. Each stage retains the newest six timestamps:

```python
stage_done_times = {
    stage: deque(maxlen=6)
    for stage in ("vision", "comfy", "vprompt", "video")
}
```

For two or more completions, realized throughput is represented as the mean
**inter-completion interval**:

```text
stage_interval = (newest_completion - oldest_completion)
                 / (number_of_samples - 1)
```

The estimator deliberately does not time one item and divide by the configured
worker count. Worker count is only a ceiling: with two jobs left, four prompt
workers do not create four-way throughput. Inter-completion timing measures the
throughput actually achieved and automatically incorporates uneven jobs,
partial worker utilization, and the video co-op.

At least two completions are required because one timestamp contains no
interval. Until then the UI honestly displays `estimating…` rather than
inventing a number. Timestamps are cleared at the beginning of each run and are
not seeded with run start time, because doing so would incorrectly include
upstream warm-up in the first video interval.

### Remaining-work calculation

For the full Render Both path, a downstream stage must process its own queued
and active items plus everything still upstream of it. Reimagine Animator takes
one locked snapshot and accumulates work in stage order:

```python
carried = 0
for stage in ("vision", "comfy", "vprompt", "video"):
    carried += pending[stage] + active_count(stage)
    remaining[stage] = carried
```

For example, if two shots are in vision, one is generating an image, and three
are waiting for video, then the video stage has six future/current items to
absorb—not merely the three currently in its queue.

This is why `total - done` is not used. Cache modes seed jobs into different
entry queues, so the run total is not the number of jobs every stage will see.
Pending counts are increased only when a job is actually seeded into or
forwarded to a stage, which also lets the snapshot reflect mid-run injection.

For a general-purpose port, improve this further by counting only jobs whose
`terminal_stage` is at or beyond the stage being estimated. The cumulative
shortcut assumes the normal forward path. Image-only work does not become video
work, and a failed or intentionally short-circuited job should be removed from
future-stage demand.

With two video workers, `active_count(video)` must count both active job IDs.
Reimagine Animator's UI stores them as a comma-separated display string and
splits it for the count; a port should keep the underlying set and count it
directly.

### Stage and overall ETA

Each stage ETA is:

```text
stage_eta = remaining_items_for_stage * mean_inter_completion_interval
```

The overall ETA is the maximum available stage ETA:

```text
overall_eta = max(stage_eta values with enough samples)
```

The maximum represents the current bottleneck. Because the video remaining
count already includes work upstream of video, the estimate only omits the last
item's remaining upstream traversal. In this workload that is usually small
relative to multi-minute video rendering.

The UI shows both a relative estimate and a wall-clock projection, such as:

```text
~11m left · done ≈14:47
```

The queue table shows `Stage`, `Active File`, `Queued`, `Done`, `ETA`, and
`Note`. The stage with the largest raw pending queue is labeled as a possible
bottleneck, while the video row shows when the 4090 co-op is engaged.

### Throughput changes when co-op changes

Starting or retiring the 4090 co-op roughly doubles or halves video throughput.
Keeping a six-sample window from the previous configuration would leave the ETA
biased for many minutes. Whenever co-op engages or retires, Reimagine Animator
keeps only the newest video completion timestamp. One more video completion
then establishes a fresh interval for the new GPU configuration.

The same reset rule should be applied whenever a port changes video concurrency,
backend, resolution, model, or another parameter that materially changes stage
throughput.

### ETA limitations and recommended improvements

The ETA is an adaptive operational estimate, not a schedule guarantee. It reacts
after completions and can move when clip complexity changes. For another project:

- Use job destinations rather than assuming every job reaches every stage.
- Keep success, failure, cancellation, and retry counts separately.
- Remove terminal failures from downstream demand immediately.
- Consider separate rate windows by resolution, duration, model, or GPU if jobs
  vary greatly.
- Reset a rate window after concurrency or backend changes.
- Use monotonic time for intervals and wall-clock time only for display.
- Take status and timestamp snapshots under one lock, then format outside it.
- Avoid a long historic average; it adapts too slowly to a newly engaged GPU.

## Portable supervisor outline

```python
async def run_pipeline(initial_jobs):
    reset_run_state()
    hard_stop.clear()

    q_v, q_c, q_vp, q_video = (asyncio.Queue() for _ in range(4))
    llm_slots = asyncio.Semaphore(4)

    seed_jobs_at_earliest_required_stage(initial_jobs)

    tasks = [
        *(vision_worker(q_v, q_c, llm_slots) for _ in range(4)),
        image_worker(q_c, q_vp, gpu="4090"),
        *(video_prompt_worker(q_vp, q_video, llm_slots) for _ in range(4)),
        video_worker(q_video, gpu="5090"),
    ]
    tasks = [asyncio.create_task(task) for task in tasks]

    coop = asyncio.create_task(
        supervise_4090_coop(q_v, q_c, q_vp, q_video, tasks)
    )

    await wait_for_debounced_idle_or_stop()
    if hard_stop.is_set():
        drain_pending_queues()
        await cancel_owned_backend_jobs()

    coop.cancel()
    send_one_sentinel_per_possible_worker()
    await asyncio.gather(*tasks, return_exceptions=True)
    publish_finished_or_stopped()
```

Treat this as architecture pseudocode, not drop-in code. Real implementations
must put all status transitions in `try/finally`, account for service timeouts,
and ensure a worker crash cannot strand a pending count or queue `task_done()`.

## Porting checklist

### Resource and service setup

- [ ] Identify which stages can overlap safely.
- [ ] Run each physical GPU's normal workload with one worker unless the
      backend explicitly supports concurrency.
- [ ] Create four LLM worker tasks for each prompt stage, but share one
      four-slot semaphore across all LM calls.
- [ ] Move blocking SDK, HTTP, filesystem, and subprocess calls off the event
      loop.
- [ ] Confirm both video backends accept equivalent job inputs and produce a
      stable output contract.
- [ ] Use unique output names across concurrent renderers.
- [ ] Measure VRAM, RAM, and system commit at peak dual-video load.

### Queue correctness

- [ ] Give each stage its own queue.
- [ ] Define a typed job and stable job ID.
- [ ] Route cached jobs to their earliest required stage.
- [ ] Update downstream pending only when forwarding succeeds.
- [ ] Always call `task_done()` in `finally`.
- [ ] Keep active jobs in sets, not single strings.
- [ ] Require queues empty **and** status idle before declaring completion.
- [ ] Debounce idle detection to cover handoff gaps.
- [ ] Perform cross-thread queue mutations with the event loop's thread-safe
      scheduling method.

### Co-op correctness

- [ ] Engage only after every upstream status and queue is empty.
- [ ] Require enough queued videos to justify model switching.
- [ ] Free image and LLM models before loading a second video model.
- [ ] Consume one shared video queue from both GPUs.
- [ ] Retire the 4090 with a private event, not a shared sentinel.
- [ ] Wait for the old co-op worker to exit before starting another.
- [ ] Reset video ETA samples whenever the worker count changes.
- [ ] Disable co-op for incompatible backends or mutually exclusive power mode.

### Stop and recovery

- [ ] Make Stop cancel queued and active pipeline-owned generation.
- [ ] Check Stop after dequeue and again before forwarding.
- [ ] Track owned backend job IDs per service instance.
- [ ] Keep a `stopping` phase until workers and backend cancellation finish.
- [ ] Prevent new runs and destructive project actions while stopping.
- [ ] Send one shutdown sentinel per worker that could still be awaiting a
      queue.
- [ ] Log cancellation failures instead of falsely claiming a confirmed stop.

### ETA and observability

- [ ] Record monotonic completion timestamps at the single `done` update path.
- [ ] Require two timestamps before showing an ETA.
- [ ] Use inter-completion intervals, not duration divided by configured workers.
- [ ] Include active plus pending upstream work in downstream demand.
- [ ] Account for each job's terminal stage.
- [ ] Reset rate samples after material throughput changes.
- [ ] Display active, queued, done, ETA, and co-op state per stage.

## Tests to carry into the new project

At minimum, automate these scenarios:

1. Four LLM calls can overlap, while a fifth waits for the shared semaphore.
2. Vision and video-prompt pools together never exceed four LM calls.
3. The first video starts before the final image completes.
4. The 4090 co-op does not start while any upstream queue or worker is active.
5. Co-op starts when upstream is drained and at least two videos are queued.
6. A mid-run upstream injection retires co-op after its in-flight clip.
7. Co-op cannot re-engage until the retired worker has exited.
8. Two video workers never choose the same output path.
9. One video worker finishing does not hide the other active worker.
10. Natural completion waits for all queues and active sets to remain empty.
11. Hard Stop drains queued work, cancels both GPUs, and forwards no late result.
12. Cached jobs enter the correct downstream queue.
13. ETA stays `estimating` until two completions exist.
14. ETA remaining-work counts cached entry points and mid-run injections.
15. Engaging and retiring co-op resets the video throughput window.
16. Images-only jobs are not counted as future video work in the ported model.

## Load-bearing rules summary

- Four prompt workers means a **shared limit of four total LLM calls**, not four
  per prompt stage.
- Let the 5090 begin video immediately; only lend the 4090 after all upstream
  image and prompt work has drained.
- Both video GPUs pull from one queue so faster hardware naturally takes more
  work.
- Use active-job sets and unique output names when a stage has multiple workers.
- Retire a temporary worker with its own event, never a shared FIFO sentinel.
- Keep the event loop non-blocking and perform queue mutations on its thread.
- Stop must drain queued jobs, cancel owned backend work, suppress late
  forwarding, and wait for teardown before reporting success.
- Estimate throughput from real completion spacing and reset the sample window
  whenever concurrency changes.

Those rules provide the transferable speedup: continuous pipeline overlap,
bounded LLM parallelism, early use of the 5090, and safe reuse of the 4090 for
the long video tail.
