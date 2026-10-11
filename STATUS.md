# LocalRecall Status

## Environment Preflight

Initial preflight run from `/home/asus/Documents/hackathon/hk` on 2026-10-10. Commands and observations:

| Check | Result |
|---|---|
| `uname -a`, `uname -m` | Linux `4ES5`, kernel `6.17.0-1014-nvidia`, `aarch64` |
| `/etc/os-release` | Ubuntu 24.04.4 LTS (Noble) |
| `python3 --version` | Python 3.12.3 |
| `nvidia-smi` | NVIDIA GB10; driver 580.142; CUDA 13.0; command succeeded |
| `free -h` | 121 GiB total, 116 GiB available at inspection |
| `df -h .` | Workspace filesystem: 916 GiB total, 617 GiB available |
| `docker --version` | Docker 29.2.1 |
| `docker info` | **Failed:** permission denied opening `/var/run/docker.sock`; current user is not in the `docker` group |
| Hugging Face cache lookup | No `*Qwen3.8-27B-NVFP4*` directory found under `~/.cache/huggingface/hub`; inspected cache is only 40 KiB; `hf` CLI is absent |
| `hostname -I` | Two addresses returned; exact host addresses are omitted from tracked status |
| `ss -ltn` | SSH listens on `0.0.0.0:22`; no LocalRecall ports 7860 or 8000 were listening |
| Host display state | `nvidia-smi` lists Xorg, GNOME Shell, and Firefox; this host is not currently operating as the headless-server setup described in the runbook |
| Repository at initial inspection | `main` at `f17f59d`; the planning documents and outdated README were present |

Initial interpretation: this is an ARM64 Ubuntu GB10 environment with a suitable Python version and GPU, but the model/container gate was not yet verified. At that time Docker socket access blocked vLLM in Docker and the checkpoint was absent. The workspace filesystem reports 916 GiB, not the 4 TB machine assumption in the planning docs. The host appears to have a desktop, while the application design still correctly keeps camera capture in the presenter laptop browser.

## Milestones

### Documentation and preflight baseline

- Completed: read the pasted task and all four authoritative documents fully.
- Completed: rewrote `README.md` to reflect the headless-browser camera architecture, Dockerized vLLM preference, local-only inference, privacy caveats, and honest implementation status.
- Completed: recorded the actual environment preflight above.
- Completed: added `scripts/preflight.sh` and ran it successfully; it reports Docker/cache blockers without failing the remaining environment checks.
- Commands run: `./scripts/preflight.sh` (completed; Docker inaccessible, Qwen cache absent); `uname -a`; `uname -m`; `cat /etc/os-release`; `python3 --version`; `docker --version`; `docker info`; `nvidia-smi`; `free -h`; `df -h .`; `hostname -I`; `ss -ltn`; Hugging Face cache `find` and `du`; `git status`, `git diff`, `git log`.
- Tests: preflight script execution succeeded. The current application test suite is recorded under the manual mock vertical slice below.
- Blockers at this milestone: Docker daemon socket permission denied. Need determine whether this is the organizer GX10 or development Spark before marking target-hardware validation.
- Next: resolve Docker access and model acquisition path; then inspect the vLLM image entrypoint/flags, start Qwen in Docker, and pass text and image smoke tests before proceeding to application implementation. [NVIDIA's DGX Spark vLLM playbook](https://build.nvidia.com/spark/vllm/choose-recipe) identifies Qwen3.8-27B NVFP4 as the one-Spark recipe and links `vllm/vllm-openai:qwen38`; the [linked vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=dgx_spark_gb10) currently says text serving is what it has verified, so image support remains a required local smoke-test gate.

### Model-gate preparation

- Completed: created pinned `pyproject.toml`, `.gitignore`, `scripts/smoke_text.py`, and `scripts/smoke_vision.py`. The vision smoke script generates a labeled synthetic scene pair/composite, sends an image question and a left-to-right event question to the local endpoint, validates the response shape/direction, and saves inputs and raw responses under `data/fixtures/smoke/`.
- Commands run: `python3 -m py_compile scripts/smoke_text.py scripts/smoke_vision.py`; `bash -n scripts/preflight.sh`; `git diff --check` (all succeeded). `hf download --dry-run` succeeded unauthenticated and found 19 model files totaling 21.9 GB. Started actual checkpoint transfer with `hf download ... --max-workers 4`; it was stopped at about 4.4 GB at the user's request. The Qwen cache entry was later removed with `hf cache rm`.
- Tests: the text and vision smoke scripts returned the expected output against the explicit loopback mock server; this validates only protocol shape and parsing. Actual Qwen inference is not yet run.
- At that milestone, Docker was not accessible and the checkpoint transfer had stopped at about 4.4 GB. Both facts were updated later: passwordless sudo now reaches Docker, and the partial Qwen cache has been removed.
- Next: poll the same download session until terminal, verify checkpoint files, obtain Docker access, inspect the exact image entrypoint and supported arguments, then start vLLM and run both smoke tests.

### Manual mock vertical slice

- Implemented: pinned Python dependencies; shared Pydantic records and facade contract; local-only configuration validation; SQLite WAL schema with foreign keys, transactional event/state updates, rebuild, approval/rejection, and deletion; labeled BEFORE/AFTER composite generation; local OpenAI-compatible VLM client and deterministic mock; one-repair event validation; manual `analyze_pair` facade with one bounded inference worker; alias-based deterministic query path; browser frame ring/lease ingress; fixture-video file decoder; bounded transient-cache cleanup; mock facade; five-tab Gradio UI; Docker vLLM launcher with runtime entrypoint/help/flag checks; text/vision smoke scripts.
- Commit: `9fe0b4c feat: scaffold mock-backed LocalRecall manual pipeline`.
- Verification: `.venv/bin/python -m pip install -e '.[dev]'` completed successfully. `make test` passed **16 tests**. `.venv/bin/ruff check app scripts tests` passed. Launched `.venv/bin/python -m app.ui.app --mock --host 127.0.0.1 --port 7861`; `GET /` and `GET /config` returned HTTP 200, with Watch, Memory, Ask, Setup, and System & Privacy. The UI process was stopped after the check. A loopback OpenAI-compatible mock server exercised text and image request scripts and the facade HTTP integration; these tests did not run Qwen. Verified Gradio 5.49.1 reads the configured `GRADIO_TEMP_DIR` before launch.
- Not verified: vLLM container startup, real Qwen text/vision requests, actual browser upload in a browser, automatic motion detection, WAN-off demo, and target ASUS GX10 validation. There is no `manual-mvp` tag.
- At this milestone: Docker daemon socket required authorization and the host checkpoint download was active. Both conditions changed later: passwordless sudo now works, and the partial host checkpoint was removed at the user's request.
- Next: finish and verify the Docker model gate, then close remaining manual-backend gaps and run the manual path with real Qwen before treating the manual MVP as complete.

### Evaluation and demo tooling

- Implemented: deterministic 20-trial synthetic ground-truth manifest; four object classes and three zones; generated BEFORE/AFTER pairs and optional stable/move/no-change videos; direct `EventAnalyzer` mock/real evaluation; raw per-trial response/prediction and latency JSON; Markdown metrics report; recorded-trial folder convention; offline connectivity checker; demo bundle exporter; evaluation and pitch docs.
- Commands run: `make eval`; `.venv/bin/python -m pytest` (18 passed); `.venv/bin/ruff check app scripts tests` (passed); `bash -n scripts/*.sh`; `git diff --check`.
- Mock evaluation: 20 synthetic trials; event precision 0.722, recall 1.000, false positives 5, event-type accuracy 50.0%, object accuracy 27.8%, source/destination zone accuracy 37.5%. The deterministic mock always predicts a move for its first listed object from first to last zone. Its median pair-to-parse harness time was 17.412 ms; real model latency is not measured. This is a harness check, not Qwen or product accuracy.
- Not measured: Qwen inference, real recorded-trial accuracy, continuous-video persisted bytes, real end-to-end latency, WAN-off inference, browser-camera use, and actual GX10 identity.
- Model weights: the partial Hugging Face repository cache was removed with `hf cache rm model/nvidia/Qwen3.8-27B-NVFP4 --yes`; the tool reported 2.0 GB freed and the remaining HF hub cache was 68 KB. No Qwen weights are currently available on the host or in a Docker volume.
- Docker: direct `docker` still returns permission denied for user `asus`, but the user authorized passwordless sudo and `sudo -n docker info` succeeds with Docker Engine 29.6.2. `scripts/docker_access.sh` now selects direct Docker or `sudo -n docker` for project lifecycle scripts.
- Runtime image: `sudo -n docker run --rm --gpus all vllm/vllm-openai:qwen38 vllm --version` is still pulling the runtime image (about 4.8 GB of containerd content present at the latest check; Docker does not list the image until pull completion). No persistent model container or model volume has been created and no Qwen inference has run. The image supplies vLLM; model weights must be downloaded separately.
- Current latest verification: `make test` passed **18 tests**; `.venv/bin/ruff check app scripts tests` passed; `make eval` generated the mock JSON and Markdown outputs under ignored `data/evaluation/results/`.
- `make offline-check` was exercised against the local loopback mock VLM (`127.0.0.1:8010`) and mock Gradio (`127.0.0.1:7861`); local URL and health checks passed. This did **not** disconnect WAN and is not evidence of WAN-off operation.
- Docker was retried after the user's update. Direct `docker info` is still denied because `asus` is not in the `docker` group; `sudo -n docker info` now succeeds as authorized. The `qwen38` runtime image pull remains active.
- Next: finish pulling and inspect the runtime image; store weights in the agreed Docker-managed volume; verify GPU access and local image inference; then run the manual UI flow. Keep motion-triggered Watch disabled until these gates pass.

### Backend Friend 1: model answer citation guard

- Implemented: the real VLM query-answer path now refuses to return model prose when it cites no saved event or cites any event ID outside the supplied SQLite records. An empty or malformed record list also returns the fixed no-reliable-observation response without calling the model. This prevents fabricated event IDs from being presented as evidence.
- Changed: `app/vlm_client.py`.
- Verification: not run in this milestone. The user requested implementation and explanation, not test execution. Existing project test results above are from the prior milestone and do not verify this change.
- Model readiness remains unverified: the Docker runtime image pull is still in progress, no Qwen checkpoint has been downloaded, and no real Qwen text or image inference has run.
- Next: once the model image is present, inspect the actual runtime help, download/validate the model checkpoint into the Docker-managed cache, then run the text and vision smoke tests and record actual results.

### Frontend A: first-use Gradio layout

- Implemented: made the supported manual BEFORE/AFTER image path the first and most prominent Watch-tab action; grouped its result summary, evidence gallery, and collapsed structured details; moved the optional camera stream into a collapsed section with explicit ingress-only wording; removed Start/Stop Watch buttons that did not control an automatic-analysis pipeline. Added clearer setup order, memory/review guidance, Ask help text, privacy wording, and responsive local styling.
- Changed: `app/ui/app.py` only for this milestone.
- Verification: no tests or browser launch were run in this milestone. The UI layout still needs a visual check in mock mode at laptop and narrow widths.
- Current observed model setup: Docker image `vllm/vllm-openai:qwen38` is present; Docker volume `localrecall-qwen38-cache` was not present at the latest inspection, so the first model start still needs to download weights.
- Next: view the mock UI in a browser, confirm all five tabs and image-pair flow, then adjust any layout issues found. Keep auto Watch disabled until the real local image and manual database/query gates pass.

### Frontend A: completed presentation layer

- Implemented: strengthened the first-use Gradio experience with a prominent Manual Remember callout, explicit laptop-browser camera instructions, live stale/disconnected ingress status, readable event confidence/memory-state language, evidence-backed Ask output, Setup tables for zones and registered objects, and a local/privacy explanation panel.
- Implemented: moved timeline/setup formatting into UI callbacks and presentation transforms; improved the mock facade so Ask displays fixture evidence and reports fixture evidence counts without initializing backend storage.
- Changed: `app/ui/app.py`, `app/ui/callbacks.py`, `app/ui/transforms.py`, `app/ui/mock_facade.py`, `tests/test_ui_contract.py`, and `UI_HANDOFF.md`.
- Verification: `git diff --check` passed. Automated Python tests, Ruff, and browser launch were not run because this Windows workspace has no Python interpreter, `.venv`, or WSL installation available.
- Not verified: visual browser review, `python -m app.ui.app --mock --port 7861`, `pytest tests/test_ui_contract.py`, and `ruff check app scripts tests`. These remain the next verification step on the project’s Linux/Python environment.

### Frontend B: callbacks, mock parity, and labeling

- Implemented: thin Gradio callbacks wired to `LocalRecallFacade`; mock restore after delete-all; shared enrollment/zone helpers on `app/facade.py` for mock/backend parity; `MOCK_MODE.md` labeling policy; dynamic load-demo messaging from `DEFAULT_ZONES`; Python 3.11–3.12 gate in `ensure_python.sh`.
- Changed: `app/ui/*`, `app/facade.py` (shared validators only on this PR—demo zone upsert is on `feature/backend-demo-zones`), `scripts/ensure_python.sh`, `Makefile`, `README.md`, `AGENTS.md`, `tests/test_ui_callbacks.py`, `tests/test_facade_conformance.py`, `tests/test_mock_labeling.py`.
- Verification: `make test` (70 passed) and `make lint` observed on macOS mock path; mock UI smoke on Setup/Memory/System tabs; callback smoke for Analyze/Ask/review/export/delete/restore.
- Next: open Frontend B PR stacked on merged or parallel backend demo-zone PR; optional browser BEFORE/AFTER upload QA on Watch tab.

### Integration: Frontend A + Frontend B on master, first real-Qwen end-to-end trials

- Merged: PR #3 (Frontend A) as `9c9d20d`; PR #2 (Frontend B) as `c358c5b`, with conflicts in `STATUS.md` and `app/ui/{app,callbacks,mock_facade,transforms}.py` resolved (A's layout and formatting, B's wiring, validation, and mock behavior). The `upsert_zone` path from Frontend B landed with this merge.
- Runtime observed: vLLM on `127.0.0.1:8000` answers `/health`; `/v1/models` serves id `qwen38-27b` with root `Inferact/Qwen3.8-27B-NVFP4` and `max_model_len` 65536. It was not started by `scripts/start_vllm.sh`. The app requires `VLM_MODEL=qwen38-27b` in the gitignored `.env` on this machine.
- Fixed from real-model failures: with thinking enabled, two-object scenes exhausted the 512-token budget in reasoning and returned empty content (the no-change trial used 506 of 512 tokens). Requests now send `chat_template_kwargs: {"enable_thinking": false}`. With thinking off, an occluded object was reported as `disappear` at 0.95 confidence and auto-accepted, so `event_extractor.txt` now requires `uncertain` for covered or not clearly visible objects and `appear` (never move from `outside_view`) for new objects. `tests/test_vlm_http_integration.py` now asserts the configured model name instead of a hard-coded one.
- Real trials, through `LocalRecallFacadeImpl.analyze_pair` with a temporary SQLite database, synthetic labeled scenes, and real `qwen38-27b` inference (final run, after both fixes):

  | Trial | Model result | Memory effect | Model latency |
  |---|---|---|---|
  | Move: red cube left_tray → right_tray | `move` red_cube left_tray → right_tray, 0.95, accepted | state right_tray (event #1) | 6930 ms |
  | No change: hand passes through | `no_change`, 0.95 | state unchanged | 6676 ms |
  | Newly enrolled blue mug appears in left_tray | `appear` blue_mug → left_tray, 0.95, accepted | state left_tray (event #3) | 5951 ms |
  | Ambiguous: mug occluded in AFTER | `uncertain`, 0.40 | state unchanged; no evidence saved | 5623 ms |

- Ask, deterministic path: "Where is the red cube?" → Right Tray, cites #1; "Where is the blue mug?" → Left Tray, cites #3 (unchanged by the ambiguous trial); "What moved most recently?" → #3. Every cited evidence file existed on disk and every answer matched the SQLite current state.
- Gates: `make test` 74 passed; `make lint` clean.
- Not verified: real camera photos (trials used synthetic labeled shapes), accuracy beyond these four trials, the real Gradio UI in a browser with real inference, the presenter-laptop SSH/webcam path, WAN-off operation, and the organizer GX10.
