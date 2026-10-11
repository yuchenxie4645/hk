"""LocalRecall Gradio Blocks app for local backend or isolated mock mode."""

from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

configured_temp_dir = os.environ.get("GRADIO_TEMP_DIR") or dotenv_values(".env").get("GRADIO_TEMP_DIR")
os.environ.setdefault(
    "GRADIO_TEMP_DIR",
    str(Path(configured_temp_dir or "./data/tmp/gradio").expanduser().resolve()),
)

import gradio as gr

from app.config import Settings
from app.facade import LocalRecallFacadeImpl
from app.ui import callbacks
from app.ui.mock_facade import MockFacade


def build_refresh_outputs(facade):
    """Build timer outputs while reading backend health only once."""

    return callbacks.refresh_all(facade)


def create_app(facade, *, temp_allowed_path: str | None = None) -> gr.Blocks:
    data_path = Path("./data").resolve()
    allowed_paths = [str(data_path)]
    if temp_allowed_path:
        allowed_paths.append(temp_allowed_path)
    else:
        mock_temp = getattr(facade, "_temp", None)
        if mock_temp:
            allowed_paths.append(str(Path(mock_temp).resolve()))
    css = """
    .gradio-container { max-width: 1320px !important; padding: 1.25rem !important; }
    .lr-header {
        padding: 1.5rem 1.75rem;
        border: 1px solid #dbe5f0;
        border-radius: 20px;
        background: linear-gradient(115deg, #f5f8ff 0%, #f2fbfa 100%);
        box-shadow: 0 8px 28px rgba(30, 64, 110, 0.06);
    }
    .lr-eyebrow {
        color: #46617e;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .lr-note { color: #526277; font-size: 0.96rem; line-height: 1.55; }
    .lr-callout {
        margin: 0.75rem 0;
        padding: 0.9rem 1rem;
        border-left: 4px solid #4f46e5;
        border-radius: 10px;
        background: #f7f8ff;
        color: #29364a;
        line-height: 1.5;
    }
    .lr-success {
        padding: 0.7rem 0.85rem;
        border: 1px solid #b8e0cf;
        border-radius: 10px;
        background: #f2fbf6;
    }
    .lr-privacy {
        padding: 0.85rem 1rem;
        border: 1px solid #dbe5f0;
        border-radius: 12px;
        background: #fbfcfe;
    }
    .lr-small { color: #526277; font-size: 0.86rem; line-height: 1.45; }
    .lr-section {
        margin: 0.25rem 0 0.75rem;
        padding: 0.25rem 0;
    }
    @media (max-width: 700px) {
        .gradio-container { padding: 0.65rem !important; }
        .lr-header { padding: 1rem; border-radius: 14px; }
    }
    """
    with gr.Blocks(
        title="LocalRecall",
        css=css,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate"),
    ) as demo:
        source_id = gr.State(value="")
        gr.Markdown(
            "<div class='lr-eyebrow'>Local workspace memory</div>\n"
            "# LocalRecall\n"
            "Remember where things were last seen. Add a workspace, compare two moments, "
            "and ask questions backed by saved event evidence.\n\n"
            "**SET UP** → **CAPTURE** → **REVIEW** → **ASK**\n\n"
            "Private by design: backend inference runs locally; mock mode uses fixtures and never starts a model.",
            elem_classes=["lr-header"],
        )
        status_line = gr.Markdown("Checking local status…")

        with gr.Tabs():
            with gr.Tab("Watch"):
                gr.Markdown("## Remember a change", elem_classes=["lr-section"])
                gr.Markdown(
                    "### Recommended demo path: Manual Remember\n"
                    "Add an earlier image on the left and a later image on the right. "
                    "Capture each moment with the Spark camera, or upload PNG/JPEG files as a fallback. "
                    "This path works even when live streaming is unavailable.",
                    elem_classes=["lr-callout"],
                )
                with gr.Row():
                    with gr.Column():
                        before = gr.Image(
                            label="1 · BEFORE — earlier moment",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )
                        capture_before_button = gr.Button("Capture BEFORE (Spark camera)")
                    with gr.Column():
                        after = gr.Image(
                            label="2 · AFTER — later moment",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )
                        capture_after_button = gr.Button("Capture AFTER (Spark camera)")
                analyze_button = gr.Button("Analyze BEFORE + AFTER", variant="primary", size="lg")
                with gr.Group():
                    gr.Markdown("### What LocalRecall observed")
                    event_summary = gr.Markdown(
                        "Add both images, then choose **Analyze BEFORE + AFTER**. "
                        "The result will explain what changed, how confident the system is, "
                        "and whether review is needed."
                    )
                    event_evidence = gr.Gallery(label="Images supporting this result", columns=3, height=260)
                    with gr.Accordion("Structured event details", open=False):
                        event_details = gr.JSON(label="Validated event record")

                with gr.Accordion("Optional browser camera stream · frame status only", open=False):
                    gr.Markdown(
                        "This optional stream sends laptop-browser frames to the local app and reports "
                        "whether frames arrive. Automatic movement analysis is not enabled in this build. "
                        "Use the two image boxes above to analyze a change.",
                        elem_classes=["lr-note"],
                    )
                    with gr.Row():
                        browser_camera = gr.Image(
                            label="Laptop camera preview",
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                        )
                        with gr.Column():
                            ingest_status = gr.Markdown(
                                "No browser frames received yet. Manual image analysis is available above."
                            )
                            camera_health = gr.Markdown("**Laptop browser frame status:** not started")
                            gr.Markdown(
                                "Use the preview’s own controls to start or stop the stream. "
                                "This preview uses the laptop browser camera, not the Spark camera. "
                                "This ingress-only preview reports frame arrival; it does not detect or analyze movement.",
                                elem_classes=["lr-small"],
                            )

            with gr.Tab("Memory"):
                gr.Markdown("## Saved workspace memory")
                gr.Markdown(
                    "Current locations come from accepted events. Pending events wait for your decision "
                    "and do not change an object’s saved location. Open the evidence gallery to see the "
                    "BEFORE, AFTER, and model-comparison images retained for an event.",
                    elem_classes=["lr-note"],
                )
                with gr.Row():
                    object_table = gr.Dataframe(
                        headers=["Object", "ID", "Current location", "Last event"],
                        label="Current object state",
                        interactive=False,
                    )
                    event_table = gr.Dataframe(
                        headers=[
                            "Event",
                            "Timestamp (UTC)",
                            "Object",
                            "Type",
                            "From",
                            "To",
                            "Confidence",
                            "Memory state",
                        ],
                        label="Recent event timeline",
                        interactive=False,
                    )
                review_table = gr.Dataframe(
                    headers=["Event ID", "Object", "Type", "Confidence", "From", "To"],
                    label="Events awaiting review",
                    interactive=False,
                )
                gr.Markdown("### Review an event")
                with gr.Row():
                    review_id = gr.Number(label="Event ID to review", precision=0)
                    approve = gr.Button("Approve event")
                    reject = gr.Button("Reject event")
                review_result = gr.Markdown(
                    "Choose an event ID from **Events awaiting review**, then approve or reject."
                )
                gr.Markdown(
                    "### Correct an event from visible evidence\n"
                    "Use the event ID from the review list. Corrections are applied to the saved event "
                    "and the current object state is rebuilt from accepted history.",
                    elem_classes=["lr-note"],
                )
                with gr.Row():
                    correction_id = gr.Number(label="Event ID to correct", precision=0)
                    correction_type = gr.Dropdown(
                        choices=["move", "appear", "disappear", "no_change"],
                        value="move",
                        label="Correct event type",
                    )
                    correction_object = gr.Textbox(label="Object ID", placeholder="keys")
                    correction_from = gr.Textbox(label="From zone ID", placeholder="left_tray")
                    correction_to = gr.Textbox(label="To zone ID", placeholder="right_tray")
                    correction_button = gr.Button("Apply correction")
                correction_result = gr.Markdown(
                    "Enter an event ID from the timeline, adjust fields, then apply correction."
                )
                memory_gallery = gr.Gallery(label="Saved event evidence", columns=4, height=300)

            with gr.Tab("Ask"):
                gr.Markdown("## Ask about saved observations")
                gr.Markdown(
                    "Try **Where are my keys?** Answers use saved records, cite event IDs, and show "
                    "supporting evidence when available. If no saved observation supports an answer, "
                    "LocalRecall will say so.",
                    elem_classes=["lr-note"],
                )
                question = gr.Textbox(
                    label="Ask about your workspace",
                    placeholder="Where are my keys?",
                    lines=1,
                )
                with gr.Row():
                    ask_button = gr.Button("Ask LocalRecall", variant="primary")
                    gr.Examples(
                        examples=["Where are my keys?", "What moved most recently?", "Show event #1"],
                        inputs=question,
                    )
                answer = gr.Markdown("Your evidence-backed answer will appear here.")
                ask_evidence = gr.Gallery(label="Answer evidence", columns=3, height=260)

            with gr.Tab("Setup"):
                gr.Markdown("## Prepare your workspace")
                gr.Markdown(
                    "First load or define the zones visible in your camera view. Then enroll each object "
                    "with a clear reference photo and a name you will use in questions. "
                    "Normalized coordinates run from 0 to 1; a drag-to-draw editor is not required.",
                    elem_classes=["lr-note"],
                )
                setup_source_status = gr.Markdown(
                    "**Capture source:** paired-image upload/snapshot is ready · optional laptop-browser ingress is available on Watch. "
                    "Fixture inputs are supported for tests and demos."
                )
                if isinstance(facade, MockFacade):
                    gr.Markdown(
                        "In mock mode, **Load three demo zones** registers the three default regions and "
                        "restores them if you disabled one earlier. The **Memory** tab lists **events**, not zones. "
                        "Use **Load full mock demo** below to restore sample objects and events after **Delete all memory**.",
                        elem_classes=["lr-note"],
                    )
                    load_full_mock_button = gr.Button(
                        "Load full mock demo (zones, keys, events)",
                        variant="primary",
                    )
                    load_full_mock_result = gr.Markdown()
                zones_button = gr.Button(
                    "Load three demo zones",
                    variant="secondary" if isinstance(facade, MockFacade) else "primary",
                )
                zone_result = gr.Markdown("No zones loaded yet.")
                with gr.Row():
                    setup_zone_table = gr.Dataframe(
                        headers=["Zone", "ID", "Normalized bounds", "Status"],
                        label="Configured zones",
                        interactive=False,
                    )
                    setup_object_table = gr.Dataframe(
                        headers=["Object", "ID", "Aliases", "Reference", "Status"],
                        label="Registered objects",
                        interactive=False,
                    )
                gr.Markdown("### Enroll an object")
                with gr.Row():
                    object_name = gr.Textbox(label="Object name", placeholder="Keys")
                    object_aliases = gr.Textbox(label="Aliases, separated by commas", placeholder="car keys, key ring")
                reference = gr.Image(
                    label="Object reference image · upload or browser snapshot",
                    sources=["webcam", "upload"],
                    type="numpy",
                )
                enroll_button = gr.Button("Save object reference", variant="primary")
                setup_result = gr.Markdown()
                gr.Markdown("### Add a rectangular zone")
                with gr.Row():
                    zone_name = gr.Textbox(label="Zone name", placeholder="Shelf")
                    zone_slug = gr.Textbox(label="Zone ID", placeholder="shelf")
                with gr.Row():
                    x1 = gr.Number(label="x1", value=0.0, minimum=0, maximum=1)
                    y1 = gr.Number(label="y1", value=0.0, minimum=0, maximum=1)
                    x2 = gr.Number(label="x2", value=1.0, minimum=0, maximum=1)
                    y2 = gr.Number(label="y2", value=1.0, minimum=0, maximum=1)
                zone_save = gr.Button("Save zone")
                with gr.Row():
                    delete_object_id = gr.Textbox(label="Object ID to remove")
                    delete_object_button = gr.Button("Remove object reference")
                with gr.Row():
                    delete_zone_id = gr.Textbox(label="Zone ID to disable")
                    delete_zone_button = gr.Button("Disable zone")

            with gr.Tab("System & Privacy"):
                gr.Markdown("## System health and data handling")
                system_status = gr.Markdown()
                gr.Markdown(
                    "The Spark camera takes a snapshot only when you press a Capture button. "
                    "Laptop-browser camera frames travel from this browser to the local backend over your laptop-to-GB10 connection. "
                    "The app retains enrolled object references and selected event evidence; it does not write a "
                    "continuous video. Gradio may create temporary upload/cache files in the configured local "
                    "folder, which the app manages and cleans up. The status above reports the observed cache count.",
                    elem_classes=["lr-privacy"],
                )
                with gr.Accordion("How this local demo works", open=False):
                    gr.Markdown(
                        "- A camera attached to the Spark takes one snapshot per Capture button press; it never records video. "
                        "File uploads and the laptop-browser webcam remain optional fallbacks.\n"
                        "- The manual BEFORE/AFTER path is the reliable fallback and sends one bounded comparison to the local model.\n"
                        "- Backend A validates the model response against registered object and zone IDs before memory can change.\n"
                        "- SQLite is the factual memory. Low-confidence or uncertain observations do not update locations.\n"
                        "- The app uses `nvidia/Qwen3.8-27B-NVFP4` only in backend mode; mock mode never starts a model."
                    )
                with gr.Row():
                    export_button = gr.Button("Export session")
                    export_file = gr.File(label="Session archive")
                delete_confirm = gr.Checkbox(
                    label="I understand: delete removes saved objects, zones, events, queries, and retained evidence"
                )
                delete_button = gr.Button("Delete all memory", variant="stop")
                delete_result = gr.Markdown()
                if isinstance(facade, MockFacade):
                    gr.Markdown(
                        "Mock mode keeps data in memory only. **Delete all memory** clears the demo until you "
                        "restore fixtures or restart the app.",
                        elem_classes=["lr-note"],
                    )
                    restore_mock_button = gr.Button("Restore mock demo data", variant="secondary")
                    restore_mock_result = gr.Markdown()

        def wrapped_analyze(before_image, after_image):
            try:
                return callbacks.analyze_pair(facade, before_image, after_image)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_capture_before():
            try:
                return callbacks.capture_spark_frame(facade)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_capture_after():
            try:
                return callbacks.capture_spark_frame(facade)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_ask(text):
            try:
                return callbacks.ask(facade, text or "")
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_enroll(name, aliases, image):
            try:
                return callbacks.enroll_object(facade, name or "", aliases or "", image)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_load_zones():
            try:
                return callbacks.load_zones(facade)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_ingest(frame, session_id):
            try:
                return callbacks.ingest_browser_frame(facade, frame, session_id)
            except Exception as exc:  # noqa: BLE001 - display any ingress failure in the UI
                return f"Frame rejected: {exc}"

        def wrapped_review(event_id, decision):
            try:
                return callbacks.decide_event(facade, event_id, decision)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_correction(event_id, event_type, object_id, from_zone_id, to_zone_id):
            try:
                return callbacks.correct_event(
                    facade, event_id, event_type, object_id or "", from_zone_id or "", to_zone_id or ""
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_delete_object(object_id):
            try:
                return callbacks.delete_object(facade, object_id or "")
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_delete_zone(zone_id):
            try:
                return callbacks.delete_zone(facade, zone_id or "")
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        refresh_outputs = [
            status_line,
            system_status,
            camera_health,
            setup_source_status,
            object_table,
            event_table,
            memory_gallery,
            review_table,
            setup_zone_table,
            setup_object_table,
        ]
        memory_message_outputs = [review_result, correction_result]

        def refresh_all():
            try:
                return build_refresh_outputs(facade)
            except Exception as exc:  # noqa: BLE001 - status polling must show unexpected backend errors
                message = f"Status unavailable: {exc}"
                return message, message, message, message, [], [], [], [], [], []

        def memory_panel_placeholders():
            return (
                "Choose an event ID from **Events awaiting review**, then approve or reject.",
                "Enter an event ID from the timeline, adjust fields, then apply correction.",
            )

        def wrapped_save_zone(name, slug, left, top, right, bottom):
            try:
                return callbacks.save_zone(facade, name or "", slug or "", left, top, right, bottom)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_export():
            try:
                return callbacks.export_session(facade)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def wrapped_delete(confirmed):
            try:
                return callbacks.delete_all_memory(facade, confirmed)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        analyze_button.click(
            wrapped_analyze,
            inputs=[before, after],
            outputs=[event_summary, event_details, event_evidence],
            queue=True,
        )
        capture_before_button.click(
            wrapped_capture_before,
            outputs=[before],
            queue=True,
            concurrency_id="spark_camera",
            concurrency_limit=1,
        )
        capture_after_button.click(
            wrapped_capture_after,
            outputs=[after],
            queue=True,
            concurrency_id="spark_camera",
            concurrency_limit=1,
        )
        ask_button.click(wrapped_ask, inputs=[question], outputs=[answer, ask_evidence], queue=True)
        question.submit(wrapped_ask, inputs=[question], outputs=[answer, ask_evidence], queue=True)
        enroll_button.click(
            wrapped_enroll,
            inputs=[object_name, object_aliases, reference],
            outputs=[setup_result],
            queue=True,
        ).then(refresh_all, outputs=refresh_outputs, queue=False)
        zones_button.click(wrapped_load_zones, outputs=[zone_result, setup_zone_table], queue=True).then(
            refresh_all, outputs=refresh_outputs, queue=False
        )
        zone_save.click(
            wrapped_save_zone,
            inputs=[zone_name, zone_slug, x1, y1, x2, y2],
            outputs=[zone_result],
            queue=True,
        ).then(refresh_all, outputs=refresh_outputs, queue=False)
        browser_camera.stream(
            wrapped_ingest,
            inputs=[browser_camera, source_id],
            outputs=[ingest_status],
            stream_every=0.5,
            time_limit=60,
            queue=False,
        )
        approve.click(lambda event_id: wrapped_review(event_id, "approve"), inputs=[review_id], outputs=[review_result], queue=True)
        reject.click(lambda event_id: wrapped_review(event_id, "reject"), inputs=[review_id], outputs=[review_result], queue=True)
        correction_button.click(
            wrapped_correction,
            inputs=[correction_id, correction_type, correction_object, correction_from, correction_to],
            outputs=[correction_result],
            queue=True,
        )
        delete_object_button.click(wrapped_delete_object, inputs=[delete_object_id], outputs=[setup_result], queue=True)
        delete_zone_button.click(wrapped_delete_zone, inputs=[delete_zone_id], outputs=[zone_result], queue=True).then(
            refresh_all, outputs=refresh_outputs, queue=False
        )
        export_button.click(wrapped_export, outputs=[export_file], queue=True)
        delete_button.click(wrapped_delete, inputs=[delete_confirm], outputs=[delete_result], queue=True).then(
            refresh_all, outputs=refresh_outputs, queue=False
        ).then(memory_panel_placeholders, outputs=memory_message_outputs, queue=False)

        if isinstance(facade, MockFacade):

            def wrapped_restore_mock():
                try:
                    return callbacks.restore_mock_demo(facade)
                except Exception as exc:
                    raise gr.Error(str(exc)) from exc

            def refresh_after_mock_restore(message):
                return (message, *refresh_all())

            restore_mock_button.click(
                wrapped_restore_mock,
                outputs=[restore_mock_result],
                queue=True,
            ).then(
                refresh_after_mock_restore,
                inputs=[restore_mock_result],
                outputs=[restore_mock_result, *refresh_outputs],
                queue=False,
            )

            load_full_mock_button.click(
                wrapped_restore_mock,
                outputs=[load_full_mock_result],
                queue=True,
            ).then(
                refresh_after_mock_restore,
                inputs=[load_full_mock_result],
                outputs=[load_full_mock_result, *refresh_outputs],
                queue=False,
            )

        timer = gr.Timer(1.0)
        timer.tick(refresh_all, outputs=refresh_outputs, queue=False)
        demo.load(lambda: str(uuid4()), outputs=[source_id], queue=False)
        demo.load(refresh_all, outputs=refresh_outputs, queue=False)
        demo.queue(default_concurrency_limit=2, max_size=16)

    demo.allowed_paths = allowed_paths
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalRecall Gradio interface")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="run deterministic fixture UI only")
    mode.add_argument("--backend", action="store_true", help="run the local Python backend facade")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        facade = MockFacade()
        port = args.port or 7861
        temp_allowed = str(facade._temp)
    else:
        settings = Settings(app_host=args.host, app_port=args.port or 7860, inference_mode="real")
        os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(settings.gradio_temp_dir).resolve()))
        Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)
        facade = LocalRecallFacadeImpl(settings, mock=False)
        atexit.register(facade.close)
        port = args.port or settings.app_port
        temp_allowed = str(Path(settings.data_dir).resolve())
    app = create_app(facade, temp_allowed_path=temp_allowed)
    app.launch(
        server_name=args.host,
        server_port=port,
        share=False,
        show_error=True,
        allowed_paths=app.allowed_paths,
        show_api=False,
    )


if __name__ == "__main__":
    main()
