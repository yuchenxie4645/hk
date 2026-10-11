# LocalRecall

LocalRecall is a planned private visual-memory appliance for one bounded tabletop workspace. It records semantic object events and evidence so a user can ask where an enrolled object was last seen, without sending image inference to a cloud service or retaining continuous application video.

The authoritative implementation requirements are in [LocalRecall_Codex_Master_Prompt.md](LocalRecall_Codex_Master_Prompt.md), [LocalRecall_Implementation_Blueprint.md](LocalRecall_Implementation_Blueprint.md), [LocalRecall_Team_Task_Prompts.md](LocalRecall_Team_Task_Prompts.md), and [HEADLESS_CAMERA_RUNBOOK.md](HEADLESS_CAMERA_RUNBOOK.md). Those documents define the architecture. This README summarizes them and reports implementation state; it does not replace their decisions.

For team handoffs, see the beginner-focused build directions for [Frontend A](FRONTEND_A.md), [Frontend B](FRONTEND_B.md), and the [Backend](BACKEND.md).
For a two-person, beginner-friendly backend split, see [backend_friendly.md](backend_friendly.md).

## Product flow

The first acceptance path is manual: provide separate BEFORE and AFTER images in the browser, analyze the pair with a local vision-language model, validate the structured result, store event evidence and accepted state in SQLite, then answer a query from those records. This path must work without a camera or a streaming session.

A USB camera attached to the Spark is now the primary capture device: the Watch tab's **Capture BEFORE (Spark camera)** and **Capture AFTER (Spark camera)** buttons each take one snapshot on the server (`SPARK_CAMERA_INDEX`, default `0`), and no video is recorded. File uploads and the laptop-browser webcam remain fallbacks; see [RUNBOOK.md](RUNBOOK.md#spark-camera).

Optional Watch Mode comes later. The presenter laptop browser owns the webcam and sends sampled frames to the headless GB10 over a private connection. OpenCV detects scene changes from those browser-supplied frames. The system retains a bounded in-memory ring, selects stable BEFORE and AFTER images, and sends a labeled composite to the local model only after a candidate event. The GB10 itself does not need a camera.

```text
Laptop browser (webcam snapshots, uploads, optional stream)
             |
             | SSH local port forward over private LAN
             v
Gradio Blocks on GB10 localhost:7860
             |
             +--> Python facade --> OpenCV / zones / composites
             |                         |
             |                         v
             |              local vLLM endpoint :8000
             |                         |
             +--> SQLite event log <-- validated event JSON
                         |
                         +--> evidence-backed answers
```

The guiding separation is: OpenCV decides when a scene may have changed, the VLM describes visible evidence, and SQLite owns factual memory. The model is `nvidia/Qwen3.8-27B-NVFP4`, a mixed NVFP4/FP8 multimodal checkpoint. The preferred serving path is the Spark-specific Docker image `vllm/vllm-openai:qwen38`; image inference must be proven before the rest of the application depends on it.

## Scope and privacy

- Three default zones and a small registry of three to five named objects.
- One event at a time, with confidence-based acceptance or review.
- Event evidence and registered-object references may be retained; continuous application video is not written by default.
- Browser uploads and webcam frames travel over the local laptop-to-GB10 connection. A localhost SSH forward gives the laptop browser a camera-permitted origin; frames do cross that local connection.
- Gradio may create temporary image cache files. The implementation must use and document a dedicated temporary directory and cleanup behavior, and must not claim that no transient files exist without measuring it.
- Model requests stay on loopback; no cloud inference APIs, public Gradio sharing, external database, or second web API are part of the design.
- For an offline demo, disable external WAN access while keeping the laptop-to-GB10 LAN/SSH link active.

## Requirements and intended commands

The target is Ubuntu 24.04 on an NVIDIA GB10-class system, Python 3.11 or 3.12, Docker with NVIDIA Container Toolkit, and the local Qwen checkpoint. The organizer-provided ASUS Ascent GX10 is the judging target; a personal development Spark is not a substitute.

The model commands are:

```bash
./scripts/start_vllm.sh
./scripts/wait_for_vllm.sh
python scripts/smoke_text.py
python scripts/smoke_vision.py
```

The mock UI and backend commands are:

```bash
python -m app.ui.app --mock --port 7861
python -m app.ui.app --backend --host 127.0.0.1 --port 7860
```

On the presenter laptop, forward the app port and browse to localhost:

```bash
ssh -N -L 7860:127.0.0.1:7860 <spark-user>@<spark-lan-ip>
```

Then open `http://127.0.0.1:7860` on that laptop and allow camera access if using webcam inputs. File uploads remain the fallback. See [STATUS.md](STATUS.md) for which commands have been verified against the real model and target hardware.

### Local development quick start

Requires **Python 3.11 or 3.12** (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`). If your default `python3` is outside that range, `make setup` will print the detected version and install hints. Use a 3.12 env first:

```bash
conda create -y -n localrecall python=3.12
conda activate localrecall
```

```bash
./scripts/preflight.sh
make setup
make test
python -m app.ui.app --mock --port 7861
```

Mock mode is isolated from SQLite and vLLM and labels its fixture events as MOCK. The current backend command is:

```bash
python -m app.ui.app --backend --host 127.0.0.1 --port 7860
```

The backend supports manual pair analysis and the local VLM client. Automatic motion detection is not enabled yet. Use the manual image-pair path first.

### Model setup

The selected vLLM image is `vllm/vllm-openai:qwen38`. `make model-start` pulls that runtime image, creates the persistent Docker volume `localrecall-qwen38-cache`, downloads the checkpoint from inside the container into that volume on the first run, validates its shards, and mounts the volume read-only in the serving container. The weights remain outside the runtime image, so rebuilding or updating the runtime does not duplicate the checkpoint.

Then use `./scripts/wait_for_vllm.sh` and `make model-smoke`. The launcher inspects the image's vLLM help and verifies the checkpoint inside the Docker volume before serving. Docker daemon access and network access for the first model download are prerequisites; see [STATUS.md](STATUS.md) for their current state. Real Qwen text and image inference must pass before any real-model result is reported.

## Offline operation

The model endpoint defaults to `http://127.0.0.1:8000/v1`; the application rejects non-loopback model endpoints while `OFFLINE_MODE=1`. The container uses Hugging Face and Transformers offline flags after the checkpoint is present. For a WAN-off demonstration, keep the laptop-to-GB10 LAN/SSH connection active and disable only external connectivity. The browser's frames still cross that private local connection.

## Privacy and retention

The live frame source is a bounded RAM ring. Persistent files are object reference images and accepted/reviewable event evidence. Gradio uploads and image outputs can create transient cache files. The app sets `GRADIO_TEMP_DIR` to a dedicated local directory, removes files older than 24 hours, caps it at 2 GiB, and clears it on Delete All Memory and orderly shutdown. The UI reports the last measured transient-cache file and byte counts. These controls do not mean that no transient frame files exist.

## Measured results

- Mock API contract: text and image request scripts received HTTP 200 from a loopback mock server; this verified request formatting and response parsing only.
- Automated tests: see the exact latest count and command in [STATUS.md](STATUS.md).
- Real Qwen text latency, image inference, event accuracy, and end-to-end latency: **not measured**.
- Continuous-video bytes stored: **not measured on a running application**; the current implementation has no continuous-video writer.

`make eval` regenerates a 20-pair synthetic trial set and writes mock-client JSON and Markdown reports under `data/evaluation/results/`. The current deterministic mock always predicts a move for its first object, so these metrics validate the evaluation path and make its limitations visible; they are not real model or product accuracy. `EVALUATION.md` records the exact observed numbers and the real-trial command.

## Troubleshooting

| Symptom | Action |
|---|---|
| `docker info` reports permission denied | Run Docker as a user authorized for the Docker socket. The launcher will not change host group membership or permissions. |
| Model snapshot is incomplete | Finish the Hugging Face download, then rerun `./scripts/start_vllm.sh`; it checks `config.json`, the index, and every referenced weight shard. |
| Text works but image smoke test fails | Keep the app on manual image-pair mode; inspect container logs and image request support before using the VLM for application events. The authoritative plan allows trying the Spark-compatible SGLang path while preserving the `VLMClient` interface. |
| Laptop webcam is unavailable | Open the app at `http://127.0.0.1:7860` on the laptop through SSH port forwarding, grant browser permission, or upload two images. The GB10 has no required camera. |
| Camera stream stops | The browser owns the camera. Reopen the tab or use manual BEFORE/AFTER images; stale frames should leave manual analysis and Ask available. |

## Current implementation status

The environment preflight is complete and recorded in [STATUS.md](STATUS.md). The manual mock-backed facade, SQLite memory, composite builder, event validation, browser frame ingress, privacy janitor, and five-tab Gradio UI are implemented. The mock-mode UI returns HTTP 200 and exposes all five tabs. The partial Qwen checkpoint was removed from the Hugging Face cache at the user's request. Passwordless `sudo docker` reaches Docker Engine 29.6.2, and the `vllm/vllm-openai:qwen38` image pull is in progress. That image supplies the serving runtime, not Qwen weights; no real model inference result is claimed.

## Limitations

The intended MVP is a fixed, bounded workspace with enrolled objects and named zones. It is not general room surveillance or unconstrained object tracking. Occlusion, ambiguous identities, changed camera placement, lighting, and mirrored webcam images can reduce reliability. Low-confidence and invalid model outputs must not silently mutate object state. Manual BEFORE/AFTER analysis remains a first-class fallback if streaming is unavailable or unreliable.
