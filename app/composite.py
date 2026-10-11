"""Build evidence images sent to the local vision-language model."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas import ObjectRecord, ZoneRecord


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _as_rgb_image(frame: np.ndarray) -> Image.Image:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frames must be uint8 NumPy arrays")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frames must be nonempty HxWx3 RGB arrays")
    return Image.fromarray(frame)


def normalized_polygon_pixels(
    polygon: list[tuple[float, float]], width: int, height: int
) -> list[tuple[int, int]]:
    """Map normalized zone points to pixels using the inference-composite convention."""
    return [(int(x * width), int(y * height)) for x, y in polygon]


def _draw_zones(
    image: Image.Image,
    zones: list[ZoneRecord],
    *,
    draft_zone: ZoneRecord | None = None,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    palette = [(30, 115, 201), (56, 142, 60), (207, 84, 48), (133, 86, 181)]
    for index, zone in enumerate(zones):
        color = palette[index % len(palette)]
        points = normalized_polygon_pixels(zone.polygon, w, h)
        draw.polygon(points, fill=(*color, 24), outline=(*color, 230), width=max(3, w // 320))
        x, y = points[0]
        label = f"{zone.display_name} · {zone.id}"
        text_box = draw.textbbox((0, 0), label, font=_font(16, True))
        label_width = min(w - max(0, x + 4), text_box[2] - text_box[0] + 20)
        draw.rounded_rectangle(
            (x + 4, y + 4, x + 4 + label_width, y + 35),
            radius=6,
            fill=(*color, 220),
        )
        draw.text((x + 10, y + 8), label, fill="white", font=_font(16, True))
    if draft_zone is not None:
        points = normalized_polygon_pixels(draft_zone.polygon, w, h)
        color = (245, 158, 11)
        width = max(5, w // 180)
        draw.polygon(points, fill=(*color, 42), outline=(255, 255, 255, 255), width=width + 3)
        draw.line(points + [points[0]], fill=(*color, 255), width=width, joint="curve")
        x, y = points[0]
        label = f"DRAFT · {draft_zone.display_name} · {draft_zone.id}"
        text_box = draw.textbbox((0, 0), label, font=_font(16, True))
        label_width = min(w - max(0, x + 4), text_box[2] - text_box[0] + 20)
        draw.rounded_rectangle(
            (x + 4, y + 4, x + 4 + label_width, y + 35),
            radius=6,
            fill=(*color, 245),
        )
        draw.text((x + 10, y + 8), label, fill="black", font=_font(16, True))


def render_zone_overlay(
    frame_rgb: np.ndarray,
    zones: list[ZoneRecord],
    *,
    draft_zone: ZoneRecord | None = None,
) -> np.ndarray:
    """Render saved zones and an optional draft on a copy of a frozen RGB frame."""
    image = _as_rgb_image(frame_rgb.copy())
    _draw_zones(image, zones, draft_zone=draft_zone)
    return np.asarray(image, dtype=np.uint8).copy()


def build_composite(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    zones: list[ZoneRecord],
    objects: list[ObjectRecord],
    output_path: str | Path,
    *,
    reference_images: dict[str, np.ndarray] | None = None,
    before_time: datetime | None = None,
    after_time: datetime | None = None,
) -> Path:
    """Create and save one labeled BEFORE/AFTER composite at a bounded size."""
    before = _as_rgb_image(before_rgb)
    after = _as_rgb_image(after_rgb)
    target_w, target_h = 640, 400
    before.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    after.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    panel_w = max(before.width, after.width)
    panel_h = max(before.height, after.height)
    heading_h, strip_h, footer_h = 62, 132, 86
    composite = Image.new("RGB", (panel_w * 2, panel_h + heading_h + strip_h + footer_h), "#f4f2ec")
    draw = ImageDraw.Draw(composite)
    analysis_time = after_time or before_time or datetime.now(UTC)
    draw.text((18, 8), "BEFORE", fill="#14202c", font=_font(34, True))
    draw.text((panel_w + 18, 8), "AFTER", fill="#14202c", font=_font(34, True))
    draw.text(
        (18, 43),
        f"PAIR RECEIVED {analysis_time.astimezone(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        fill="#34495e",
        font=_font(12),
    )
    _draw_zones(before, zones)
    _draw_zones(after, zones)
    composite.paste(before, (0, heading_h))
    composite.paste(after, (panel_w, heading_h))

    strip_y = heading_h + panel_h
    draw.rectangle((0, strip_y, composite.width, strip_y + strip_h), fill="#ffffff")
    draw.text((18, strip_y + 8), "REGISTERED OBJECT REFERENCES", fill="#253444", font=_font(18, True))
    x = 18
    for record in objects[:5]:
        tile = Image.new("RGB", (105, 76), "#e5e7eb")
        reference = (reference_images or {}).get(record.id)
        if reference is not None:
            try:
                thumb = _as_rgb_image(reference)
                thumb.thumbnail((100, 70), Image.Resampling.LANCZOS)
                tile.paste(thumb, ((105 - thumb.width) // 2, (76 - thumb.height) // 2))
            except ValueError:
                pass
        composite.paste(tile, (x, strip_y + 36))
        draw.text((x + 112, strip_y + 39), record.id, fill="#111827", font=_font(15, True))
        draw.text((x + 112, strip_y + 59), record.display_name[:22], fill="#374151", font=_font(13))
        x += 280

    footer_y = strip_y + strip_h
    draw.rectangle((0, footer_y, composite.width, composite.height), fill="#e9edf2")
    object_ids = ", ".join(record.id for record in objects) or "none"
    zone_ids = ", ".join(zone.id for zone in zones) or "none"
    draw.text((18, footer_y + 12), f"Allowed object IDs: {object_ids}", fill="#1f2937", font=_font(15, True))
    draw.text((18, footer_y + 43), f"Allowed zone IDs: {zone_ids}", fill="#1f2937", font=_font(15, True))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(path, format="JPEG", quality=88, optimize=True)
    return path

