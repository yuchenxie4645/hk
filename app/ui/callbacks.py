"""Thin Gradio callback adapters; all application behavior remains in the facade."""

from __future__ import annotations

import numpy as np

from app.composite import render_zone_overlay
from app.contracts import LocalRecallFacade
from app.facade import DEFAULT_ZONES, build_zone_record
from app.ui.transforms import (
    event_markdown,
    event_rows,
    is_mock_labeled_model,
    object_rows,
    query_markdown,
    source_status_markdown,
    status_markdown,
    zone_rows,
)

NEW_ZONE_VALUE = "__new_zone__"


def analyze_pair(facade: LocalRecallFacade, before: np.ndarray | None, after: np.ndarray | None):
    if before is None and after is None:
        raise ValueError("Add a BEFORE image and an AFTER image, then press Analyze.")
    if before is None:
        raise ValueError("Add the BEFORE (earlier) image on the left.")
    if after is None:
        raise ValueError("Add the AFTER (later) image on the right.")
    event = facade.analyze_pair(before, after)
    gallery = [
        (path, label)
        for path, label in (
            (event.pre_image_path, "BEFORE evidence"),
            (event.post_image_path, "AFTER evidence"),
            (event.composite_path, "Model comparison"),
        )
        if path
    ]
    return event_markdown(event), event.model_dump(mode="json"), gallery


def capture_spark_frame(facade: LocalRecallFacade) -> np.ndarray:
    return facade.capture_camera_frame()


def capture_calibration_image(facade: LocalRecallFacade):
    frame = facade.capture_camera_frame()
    preview = render_zone_overlay(frame, facade.get_zones())
    return (
        frame,
        preview,
        (
            "**Calibration image captured.** Saved active zones are overlaid; the frozen base image "
            "is unchanged until you capture again."
        ),
    )


def zone_configuration_status(facade: LocalRecallFacade) -> str:
    zones = facade.get_zones()
    if not zones:
        return "No active zones configured yet. Choose **New zone** below or load the demo zones."
    return f"**{len(zones)} active zone{'s' if len(zones) != 1 else ''} configured.** Select one below to edit it."


def zone_editor_choices(facade: LocalRecallFacade) -> list[tuple[str, str]]:
    return [("New zone", NEW_ZONE_VALUE), *[(f"{z.display_name} · {z.id}", z.id) for z in facade.get_zones()]]


def select_zone(facade: LocalRecallFacade, selected_zone_id: str | None):
    selected = selected_zone_id or NEW_ZONE_VALUE
    if selected == NEW_ZONE_VALUE:
        return "", "", 0.0, 0.0, 1.0, 1.0, "Creating a new zone. Choose a unique lowercase ID."
    zone = next((item for item in facade.get_zones() if item.id == selected), None)
    if zone is None:
        raise ValueError(f"No active zone has ID `{selected}`; refresh the selection.")
    xs = [point[0] for point in zone.polygon]
    ys = [point[1] for point in zone.polygon]
    return (
        zone.display_name,
        zone.id,
        min(xs),
        min(ys),
        max(xs),
        max(ys),
        f"Editing **{zone.display_name}**. Its ID is fixed so saved event history remains valid.",
    )


def _zone_numbers(x1, y1, x2, y2) -> tuple[float, float, float, float]:
    try:
        return float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter numeric bounds for Left, Top, Right, and Bottom.") from exc


def _draft_zone(display_name, zone_id, x1, y1, x2, y2):
    name = (display_name or "").strip()
    slug = (zone_id or "").strip()
    if not name:
        raise ValueError("Enter a zone name (for example: Shelf).")
    if not slug:
        raise ValueError("Enter a zone ID (for example: shelf).")
    left, top, right, bottom = _zone_numbers(x1, y1, x2, y2)
    return build_zone_record(
        name,
        slug,
        left,
        top,
        right,
        bottom,
        active_zone_count=0,
        enforce_zone_limit=False,
    )


def preview_zone_changes(
    facade: LocalRecallFacade,
    calibration_frame: np.ndarray | None,
    display_name: str,
    zone_id: str,
    x1,
    y1,
    x2,
    y2,
):
    if calibration_frame is None:
        return None, (
            "Capture a calibration image first. Previewing never runs model inference, captures "
            "another frame, or writes to SQLite."
        )
    try:
        draft = _draft_zone(display_name, zone_id, x1, y1, x2, y2)
    except ValueError as exc:
        return render_zone_overlay(calibration_frame, facade.get_zones()), f"**Cannot preview:** {exc}"
    return (
        render_zone_overlay(calibration_frame, facade.get_zones(), draft_zone=draft),
        f"**Draft preview:** {draft.display_name} (`{draft.id}`) is highlighted in amber. Nothing has been saved.",
    )


def ask(facade: LocalRecallFacade, question: str):
    text = (question or "").strip()
    if not text:
        raise ValueError("Type a question first, for example: Where are my keys?")
    result = facade.ask(text)
    if result.event_ids:
        cited = ", ".join(f"#{event_id}" for event_id in result.event_ids)
        gallery = [(path, f"Evidence for event(s) {cited}") for path in result.evidence_paths]
    else:
        gallery = []
    return query_markdown(result), gallery


def load_zones(facade: LocalRecallFacade):
    zones = facade.load_demo_zones()
    if not zones:
        return "No demo zones are available. Check backend or mock fixture setup.", []
    demo_ids = {zone.id for zone in DEFAULT_ZONES}
    loaded_defaults = [zone for zone in zones if zone.id in demo_ids]
    count = len(loaded_defaults)
    names = ", ".join(zone.display_name for zone in loaded_defaults) or ", ".join(
        zone.display_name for zone in zones
    )
    region_word = "region" if count == 1 else "regions"
    message = (
        f"**Loaded {count} default workspace {region_word}:** {names}.\n\n"
        "This re-enables any previously disabled default zones and restores their standard shapes. "
        "Check the **Configured zones** table below. Regions are not listed on the **Memory** tab "
        "(that tab is for **events** only)."
    )
    return message, zone_rows(facade.get_zones())


def enroll_object(facade: LocalRecallFacade, name: str, aliases: str, image: np.ndarray | None):
    display_name = (name or "").strip()
    if not display_name:
        raise ValueError("Enter an object name (for example: Keys).")
    if image is None:
        raise ValueError("Choose a reference image for the object.")
    record = facade.enroll_object(display_name, (aliases or "").strip(), image)
    return f"Enrolled **{record.display_name}** as `{record.id}`."


def save_zone_message(record) -> str:
    return f"Saved zone **{record.display_name}** (`{record.id}`)."


def add_zone(
    facade: LocalRecallFacade,
    display_name: str,
    zone_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    name = (display_name or "").strip()
    slug = (zone_id or "").strip()
    if not name:
        raise ValueError("Enter a zone name (for example: Shelf).")
    if not slug:
        raise ValueError("Enter a zone ID (for example: shelf).")
    left, top, right, bottom = _zone_numbers(x1, y1, x2, y2)
    return facade.add_zone(name, slug, left, top, right, bottom)


def save_zone(
    facade: LocalRecallFacade,
    display_name: str,
    zone_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> str:
    record = add_zone(facade, display_name, zone_id, x1, y1, x2, y2)
    return save_zone_message(record)


def save_zone_changes(
    facade: LocalRecallFacade,
    selected_zone_id: str | None,
    display_name: str,
    zone_id: str,
    x1,
    y1,
    x2,
    y2,
) -> str:
    selected = selected_zone_id or NEW_ZONE_VALUE
    draft = _draft_zone(display_name, zone_id, x1, y1, x2, y2)
    if selected == NEW_ZONE_VALUE:
        record = facade.add_zone(
            draft.display_name,
            draft.id,
            draft.polygon[0][0],
            draft.polygon[0][1],
            draft.polygon[2][0],
            draft.polygon[2][1],
        )
        return f"Created zone **{record.display_name}** (`{record.id}`)."
    if draft.id != selected:
        raise ValueError(
            f"Zone ID `{selected}` is fixed because saved events may reference it; "
            "choose New zone to create a different ID."
        )
    record = facade.update_zone(
        draft.display_name,
        selected,
        draft.polygon[0][0],
        draft.polygon[0][1],
        draft.polygon[2][0],
        draft.polygon[2][1],
    )
    return f"Updated zone **{record.display_name}** (`{record.id}`) without changing its history."

def ingest_browser_frame(facade: LocalRecallFacade, frame: np.ndarray | None, source_id: str):
    if frame is None:
        return "Waiting for the laptop browser camera to start."
    receipt = facade.ingest_frame(frame, source_id)
    if not receipt.get("accepted"):
        return f"Frame rejected: {receipt.get('reason', 'unknown reason')}"
    return f"Frame received · {receipt['frames_received']} total · {receipt['ingest_fps']} fps"


def refresh(facade: LocalRecallFacade, *, status=None):
    status = status or facade.get_status()
    objects = facade.get_objects()
    state_rows = {row["object_id"]: row for row in facade.get_current_state()}
    events = facade.get_events(12)
    rows = []
    for obj in objects:
        match = state_rows.get(obj.id)
        rows.append(
            [obj.display_name, obj.id, match["zone_name"] if match and match["zone_name"] else "Not observed", match["last_event_id"] if match else None]
        )
    gallery = []
    for event in events:
        for path, label in (
            (event.pre_image_path, "BEFORE"),
            (event.post_image_path, "AFTER"),
            (event.composite_path, "Comparison"),
        ):
            if path:
                prefix = "MOCK · " if is_mock_labeled_model(event.model_name) else ""
                gallery.append((path, f"{prefix}Event #{event.id} · {label}"))
    pending = [
        [event.id, event.object_id or "Unknown", event.event_type, event.confidence, event.from_zone_id, event.to_zone_id]
        for event in events
        if not event.accepted
    ]
    return status_markdown(status), rows, gallery, pending


def refresh_event_rows(facade: LocalRecallFacade):
    """Return the timeline rows used by the Memory tab."""

    return event_rows(facade.get_events(12))


def refresh_setup_rows(facade: LocalRecallFacade):
    """Return setup registries without leaking backend filesystem paths."""

    return zone_rows(facade.get_zones()), object_rows(facade.get_objects())


def refresh_source_status(facade: LocalRecallFacade, *, status=None):
    """Return only cheap browser-ingress health for the Watch tab."""

    return source_status_markdown(status or facade.get_status())


def refresh_all(facade: LocalRecallFacade):
    """Build every timer-refresh output from one shared system-status read."""

    status = facade.get_status()
    status_text, objects, gallery, pending = refresh(facade, status=status)
    events = refresh_event_rows(facade)
    zones, registered_objects = refresh_setup_rows(facade)
    source = refresh_source_status(facade, status=status)
    return (
        status_text,
        status_text,
        source,
        source,
        objects,
        events,
        gallery,
        pending,
        zones,
        registered_objects,
    )


def _event_id(event_id: float | None) -> int:
    if event_id is None or int(event_id) < 1:
        raise ValueError("Enter an event ID from the timeline or review list (for example: 2).")
    return int(event_id)


def decide_event(facade: LocalRecallFacade, event_id: float | None, decision: str):
    event_id = _event_id(event_id)
    try:
        event = facade.approve_event(event_id) if decision == "approve" else facade.reject_event(event_id)
    except KeyError:
        raise ValueError(f"No event #{event_id} exists. Pick an ID from the timeline or review list.") from None
    return event_markdown(event)


def correct_event(facade, event_id, event_type, object_id, from_zone_id, to_zone_id):
    event_id = _event_id(event_id)
    try:
        event = facade.correct_event(
            event_id,
            event_type,
            (object_id or "").strip() or None,
            (from_zone_id or "").strip() or None,
            (to_zone_id or "").strip() or None,
        )
    except KeyError:
        raise ValueError(f"No event #{event_id} exists. Pick an ID from the timeline or review list.") from None
    return event_markdown(event)


def delete_object(facade, object_id):
    object_id = object_id.strip()
    if not object_id:
        raise ValueError("Enter the object ID to delete.")
    try:
        facade.delete_object(object_id)
    except KeyError:
        raise ValueError(f"No enrolled object has ID `{object_id}`. Check the Memory tab's ID column.") from None
    return f"Deleted object reference `{object_id}`. Historical events remain in the timeline."


def delete_zone(facade, zone_id):
    zone_id = zone_id.strip()
    if not zone_id:
        raise ValueError("Enter the zone ID to delete.")
    if zone_id not in {zone.id for zone in facade.get_zones()}:
        raise ValueError(f"No active zone has ID `{zone_id}`. Check the Configured zones table on Setup.")
    facade.delete_zone(zone_id)
    return f"Disabled zone `{zone_id}`. Historical event evidence remains available."


def start_watch(facade: LocalRecallFacade):
    return status_markdown(facade.start_watch())


def stop_watch(facade: LocalRecallFacade):
    return status_markdown(facade.stop_watch())


def export_session(facade: LocalRecallFacade):
    path = facade.export_session()
    return str(path)


def delete_all_memory(facade: LocalRecallFacade, confirmed: bool):
    if not confirmed:
        raise ValueError("Check the confirmation box before deleting all memory.")
    facade.delete_all_memory()
    return "All enrolled objects, zones, queries, events, and retained evidence were deleted."


def restore_mock_demo(facade: LocalRecallFacade) -> str:
    restore = getattr(facade, "restore_demo_fixtures", None)
    if restore is None:
        raise RuntimeError("Restore demo data is only available in mock mode.")
    restore()
    return (
        "**MOCK · restored fixture data:** **Keys**, three zones, event **#1** (accepted), "
        "and event **#2** (pending review). Open the Memory tab to see them."
    )

