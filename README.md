# LocalRecall

**Private Visual Memory, Powered by Local AI**

LocalRecall is a privacy-preserving visual memory system built for the ASUS Local AI Track hackathon. It transforms observations of physical workspaces into structured, persistent, and searchable memories using a multimodal AI model running entirely on local hardware.

Instead of relying on a cloud service to process camera images, LocalRecall uses **Qwen3.8-27B-NVFP4**, served by vLLM on NVIDIA GB10 Grace Blackwell hardware.

## The Problem

Vision-language models can describe images, but they do not automatically maintain reliable, persistent memories of physical-world changes.

People frequently forget where they placed objects. In shared workspaces, tools and components are moved between locations, sometimes leaving no record of what happened.

LocalRecall explores a solution that remembers **where an object was last observed**, when a change was recorded, and what visual evidence supports that observation.

## How It Works

1. **Configure a workspace:** Register objects and define named spatial zones.
2. **Capture observations:** Take BEFORE and AFTER photographs using a Spark-attached camera or upload existing images.
3. **Analyze locally:** Qwen compares an annotated image pair and identifies meaningful object changes.
4. **Validate predictions:** Pydantic schemas, allowed-ID checks, and confidence thresholds determine whether an event is accepted or requires review.
5. **Store memories:** SQLite maintains historical events, photographic evidence, and last-known object locations.
6. **Ask questions:** Retrieve previous observations through natural-language questions such as "Where were my AirPods last seen?"

## Technology Stack

| Component | Technology |
|---|---|
| Hardware | NVIDIA GB10 Grace Blackwell |
| Multimodal AI | Qwen3.8-27B-NVFP4 |
| Inference engine | vLLM |
| Frontend | Gradio |
| Backend | Python |
| Camera integration | OpenCV / V4L2 |
| Data validation | Pydantic |
| Persistent memory | SQLite |
| Image processing | NumPy / Pillow |
| Query matching | RapidFuzz |

## Key Features

- On-device multimodal visual inference
- Spark-attached camera snapshot capture
- Object enrollment using reference images
- Visual workspace-zone calibration
- Structured object-movement detection
- Confidence-based event review
- Persistent last-known object locations
- Evidence-backed natural-language queries
- Local session export and data deletion
- No cloud inference API required

## Performance

Four targeted synthetic-scene tests using real Qwen inference produced the expected event outcomes, including movement, no-change, appearance, and uncertainty under occlusion.

Measured model latency ranged from **5.6 to 6.9 seconds** per event.

A separate real Gradio integration trial completed visual analysis in approximately **6.4 seconds** and answered a SQLite-backed question in **146 milliseconds**.

These measurements validate the tested workflow but do not establish general real-world accuracy.

## Engineering Challenges

Our main challenges included deploying a quantized multimodal model on GB10 hardware, preventing the model from exhausting its response budget during reasoning, and handling visually ambiguous observations.

We improved structured-output reliability by disabling unnecessary reasoning for event extraction. We also revised event interpretation to prevent a covered object from being confidently marked as disappeared.

The project required integrating a local camera, Gradio callbacks, model inference, and persistent database state while preserving clear separation between AI predictions and accepted memories.

## Privacy

The inference model and database run locally. The current application captures discrete photographs rather than recording continuous video.

Accepted and reviewable events may retain photographic evidence, while temporary files are subject to local cache-management policies.

Accessing the application from another computer can still involve private local-network traffic. Offline inference configuration does not itself prove full network isolation.

## Current Limitations

LocalRecall is a functional prototype for a bounded workspace. It currently requires manually supplied BEFORE and AFTER observations, supports a limited object registry, and records one primary event per analysis.

It cannot determine the current location of objects outside the camera's view or infer events it did not observe.

Future improvements include automatic event capture, multiple cameras, better geometric localization, calibrated uncertainty, and larger real-world evaluations.

## Our Vision

**LocalRecall turns visual observations into reusable, evidence-backed memory.**

Rather than asking a large model to repeatedly interpret the physical world, we use local multimodal inference to create structured records that can be retrieved quickly and independently of the model's context.
