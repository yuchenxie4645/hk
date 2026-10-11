# LocalRecall operator runbook

This runbook targets the headless Ubuntu 24.04 GB10 server. A USB camera attached to the Spark is the primary BEFORE/AFTER capture device; the laptop browser webcam and file uploads are fallbacks. The exact machine currently available is recorded in `STATUS.md`; do not treat development Spark testing as organizer-GX10 validation.

## One-time setup

```bash
./scripts/preflight.sh
make setup
```

Ensure Docker works using `docker info` or, on this machine, `sudo -n docker info`. The project lifecycle scripts try regular Docker first and use passwordless sudo if necessary. They do not change group membership or socket permissions.

The first `make model-start` pulls the runtime image and downloads model weights into the persistent Docker volume `localrecall-qwen38-cache`. The initial checkpoint download is about 21.9 GB. The HF CLI cache on the host is not used.

## Start the local model

```bash
./scripts/start_vllm.sh
./scripts/wait_for_vllm.sh
```

The launcher creates/populates the Docker volume, checks the model index and shards inside it, inspects the image entrypoint and `vllm serve --help`, then mounts the weights read-only for serving. It uses `vllm/vllm-openai:qwen38`, offline environment variables, `127.0.0.1:8000`, one tensor-parallel worker, 32K context, and FP8 KV cache. The model gate is not complete until the actual `make model-smoke` requests pass against Qwen, including image input and the synthetic left-to-right JSON event.

```bash
make model-smoke
```

The loopback mock server is for request-format tests only. Its output is not model evidence.

## Start the app

First use the manual pair flow. It does not depend on the browser stream.

```bash
.venv/bin/python -m app.ui.app --backend --host 127.0.0.1 --port 7860
```

For UI-only fixture work without SQLite or vLLM:

```bash
.venv/bin/python -m app.ui.app --mock --host 127.0.0.1 --port 7861
```

## Spark camera

- Set `SPARK_CAMERA_INDEX` in `.env` (default `0`, which opens `/dev/video0` through V4L2). On the development Spark, `/dev/video1` is not a capture node.
- The user running the app must be able to read `/dev/video*`. These nodes are usually `root:video`, so add the user to the `video` group (`sudo usermod -aG video "$USER"`, then log in again) unless a desktop-seat ACL already grants access. Check with `ls -l /dev/video0` and `getfacl /dev/video0`.
- To capture: on the Watch tab, press **Capture BEFORE (Spark camera)**, change the scene, press **Capture AFTER (Spark camera)**, then press **Analyze BEFORE + AFTER**. Each press opens the camera, takes one snapshot, and releases it. Snapshots are not saved unless Analyze retains them as event evidence.
- If capture fails (camera unplugged, wrong index, permission denied, or camera busy), the UI shows the error. Upload PNG/JPEG files into the same BEFORE/AFTER boxes instead.
- The organizer GX10 needs its own USB camera attached; a camera on the development Spark does not carry over.

## Connect the presenter laptop

On the laptop, keep this terminal open:

```bash
ssh -N -L 7860:127.0.0.1:7860 <spark-user>@<spark-lan-ip>
```

On that laptop, open `http://127.0.0.1:7860`. Grant the browser permission for the laptop camera if using webcam snapshots or optional streaming. Upload two image files if camera access is unavailable. Do not use a plain `http://<spark-ip>:7860` URL for webcam permission, and do not enable Gradio sharing.

The present app supports the manual pair path. The browser live stream currently reports ingress health only; automatic motion detection is not enabled. Start with Spark camera captures (or uploads) for BEFORE and AFTER, then inspect the event and ask `Where are my keys?`.

## Offline demonstration

Keep the laptop-to-GB10 local LAN/SSH link active. Disable or block only external WAN access, then repeat the local model query. Do not disconnect the only network carrying SSH. Do not modify host firewall rules automatically.

Before a WAN-isolated run, `make offline-check` verifies that the configured model URL and browser URL are loopback and checks local vLLM and Gradio responses. It cannot prove that WAN is disconnected. For that, disable only the WAN upstream while keeping the laptop-to-GB10 path active, rerun the check and image/query flow, then restore WAN as appropriate. The helper makes no firewall changes.

## Stop

Stop Gradio with Ctrl+C in its terminal. Stop vLLM without deleting the container:

```bash
./scripts/stop_vllm.sh
```

The backend configures a dedicated Gradio temp directory, prunes old files and caps its size, clears the directory on Delete All Memory, and clears it on orderly backend shutdown. Do not claim that transient uploads never exist.

## Troubleshooting

- Docker permission error: authorize the current user for the Docker socket, then start a new shell/session and rerun `docker info`.
- Model snapshot missing: complete the Hugging Face download first; the launcher refuses incomplete shards.
- Model text works but image fails: inspect vLLM logs and keep the UI on manual uploads only until image inference passes. The authoritative plan allows trying the Spark-compatible SGLang runtime without changing `VLMClient`.
- Browser webcam blocked: use localhost through the SSH `-L` tunnel or upload images.
- Stale stream: the browser owns capture; reopen it or continue using manual pair analysis and SQLite-backed Ask.
