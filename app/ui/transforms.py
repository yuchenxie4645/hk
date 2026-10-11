"""Convert facade records into compact, readable UI content.

The UI deliberately presents a plain-language explanation first. Raw records remain
available in the secondary details panel, but a user should not need to interpret
backend IDs or JSON to understand what LocalRecall observed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.schemas import EventRecord, QueryResult, SystemStatus


def is_mock_labeled_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    upper = model_name.upper()
    return upper == "MOCK" or upper.startswith(("MOCK/", "MOCK "))


def event_markdown(event: EventRecord) -> str:
    object_name = event.object_id or "Unknown object"
    transition = f"{event.from_zone_id or 'unknown'} → {event.to_zone_id or 'unknown'}"
    event_name = event.event_type.replace("_", " ").title()
    if event.accepted:
        review = "Accepted · saved to memory"
    elif event.event_type == "uncertain":
        review = "Uncertain · memory was not changed"
    else:
        review = "Needs review · memory was not changed yet"
    evidence = " ".join(event.visible_evidence) or event.uncertainty_reason or "No visible evidence supplied."
    model = event.model_name or "local model"
    latency = f" · {event.model_latency_ms} ms" if event.model_latency_ms is not None else ""
    mock_banner = ""
    if is_mock_labeled_model(event.model_name):
        mock_banner = "**MOCK · simulated event (fixture evidence)**  \n"
    return (
        f"{mock_banner}### {object_name}: {event_name}\n"
        f"**{transition}** · {event.confidence:.0%} confidence · {review} · Event #{event.id}\n\n"
        f"{evidence}\n\n"
        f"_Interpretation: {model}{latency}._"
    )


def query_markdown(result: QueryResult) -> str:
    citations = ", ".join(f"#{event_id}" for event_id in result.event_ids) or "No event citation"
    confidence = f" · **Confidence:** {result.confidence:.0%}" if result.confidence is not None else ""
    if result.event_ids:
        evidence = f"**Evidence:** {citations}{confidence}"
    else:
        evidence = "**Evidence:** No saved event supports this answer yet."
    return f"{result.answer}\n\n{evidence}"


def status_markdown(status: SystemStatus) -> str:
    if status.app_mode == "mock":
        runtime = "MOCK MODE · fixture behavior only"
    else:
        runtime = "LOCAL BACKEND · no cloud inference configured"
    model = "ready" if status.model_ready else "not ready / not checked"
    camera = source_state(status)
    age = "—" if status.seconds_since_last_frame is None else f"{status.seconds_since_last_frame:.1f}s ago"
    error = f"  \n⚠️ **Backend notice:** {status.last_error}" if status.last_error else ""
    return (
        f"**{runtime}**  \n"
        f"Offline mode: `{status.offline_mode}` · Model: `{status.model_name}` ({model})  \n"
        f"Laptop browser camera: **{camera}** · last frame: {age} · frames received: {status.frames_received}  \n"
        f"Events: {status.event_count} · evidence records: {status.evidence_count} · "
        f"temporary cache: {status.temp_cache_files} files / {status.temp_cache_bytes} bytes"
        f"{error}"
    )


def source_state(status: SystemStatus) -> str:
    """Return a user-facing source state without implying a server-side camera."""

    if status.source_connected:
        return "connected"
    if status.seconds_since_last_frame is None:
        return "not started · manual upload is ready"
    if status.seconds_since_last_frame > 4:
        return "stale / disconnected · use manual upload"
    return "waiting for browser frame"


def source_status_markdown(status: SystemStatus) -> str:
    """Format the camera transport health separately from the browser preview."""

    state = source_state(status)
    if status.source_connected:
        next_step = "Frames are arriving. This ingress-only preview does not run inference."
    elif status.seconds_since_last_frame is None:
        next_step = "Click the browser camera control to start, or use the BEFORE/AFTER upload path."
    else:
        next_step = "The browser tab may be closed or permission may be unavailable; the manual path remains ready."
    return f"**Laptop browser frame status:** {state}  \n{next_step}"


def format_event_timestamp(value: object) -> str:
    """Format an event timestamp consistently in UTC without breaking the timeline."""

    try:
        if isinstance(value, datetime):
            timestamp = value
        elif isinstance(value, str):
            timestamp = datetime.fromisoformat(value)
        else:
            return "Unknown time"
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return "Unknown time"


def event_rows(events: list[EventRecord]) -> list[list[object]]:
    """Create readable timeline rows while retaining the raw event ID."""

    return [
        [
            event.id,
            format_event_timestamp(event.started_at),
            event.object_id or "Unknown",
            event.event_type.replace("_", " "),
            event.from_zone_id or "—",
            event.to_zone_id or "—",
            f"{event.confidence:.0%}",
            "Accepted" if event.accepted else "Review / uncertain",
        ]
        for event in events
    ]


def zone_rows(zones) -> list[list[object]]:
    """Render normalized zone polygons as compact setup rows."""

    rows = []
    for zone in zones:
        xs = [point[0] for point in zone.polygon]
        ys = [point[1] for point in zone.polygon]
        bounds = f"({min(xs):.2f}, {min(ys):.2f}) → ({max(xs):.2f}, {max(ys):.2f})"
        rows.append([zone.display_name, zone.id, bounds, "Active" if zone.active else "Disabled"])
    return rows


def object_rows(objects) -> list[list[object]]:
    """Render the registered object vocabulary without exposing filesystem paths."""

    return [
        [
            obj.display_name,
            obj.id,
            ", ".join(obj.aliases) or "—",
            "Reference saved" if obj.reference_image_path else "No reference yet",
            "Active" if obj.active else "Disabled",
        ]
        for obj in objects
    ]

