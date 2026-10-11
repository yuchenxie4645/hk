# LocalRecall contributor instructions

The authoritative product and architecture requirements are in `LocalRecall_Codex_Master_Prompt.md` and `LocalRecall_Implementation_Blueprint.md`. Read them before changing behavior. `HEADLESS_CAMERA_RUNBOOK.md` is authoritative for camera/network/privacy setup. Preserve these planning documents.

## Architecture rules

- A camera attached to the Spark is the primary capture device. Open it with `cv2.VideoCapture(SPARK_CAMERA_INDEX)` only inside `SparkCameraSource` in `app/camera.py`, and only for explicit user-triggered snapshots. The laptop-browser webcam and file uploads remain optional fallbacks. Never write continuous video, and never call the VLM from capture code. This owner-approved override supersedes the camera sections of `LocalRecall_Codex_Master_Prompt.md`, `LocalRecall_Implementation_Blueprint.md`, and `HEADLESS_CAMERA_RUNBOOK.md`.
- Use Gradio Blocks and direct calls through `app/contracts.py`; do not add a separate application REST API, WebSockets, React, or a public Gradio share tunnel.
- Keep model calls behind `VLMClient`. The MVP model is `nvidia/Qwen3.8-27B-NVFP4`, served locally with the Spark-specific `vllm/vllm-openai:qwen38` Docker image.
- Keep offline model URLs on loopback. SQLite is factual memory; model prose never updates state without schema/ID validation and confidence policy.
- The manual BEFORE/AFTER path is the required first-class fallback. Do not enable automatic Watch Mode until real local image inference and the manual database/query path pass.
- Do not persist continuous video. Use the bounded browser frame ring and explicitly manage Gradio's temporary upload/cache directory.
- Never claim model accuracy, latency, offline behavior, or camera behavior without an observed test. Update `STATUS.md` after each completed milestone.

## Development commands

```bash
make setup
make test
.venv/bin/ruff check app scripts tests
python -m app.ui.app --mock --port 7861
```

Mock mode must not start a model server, access camera hardware, or initialize persistent backend storage. Keep it visibly marked MOCK; see `MOCK_MODE.md` for labeling vs validation rules.
