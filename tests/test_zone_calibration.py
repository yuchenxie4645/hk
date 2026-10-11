import numpy as np
import pytest

from app.composite import normalized_polygon_pixels, render_zone_overlay
from app.config import Settings
from app.facade import LocalRecallFacadeImpl
from app.schemas import ZoneRecord
from app.ui import callbacks
from app.ui.mock_facade import MockFacade


def test_normalized_zone_mapping_matches_composite_geometry():
    polygon = [(0.1, 0.2), (0.75, 0.2), (0.75, 1.0), (0.1, 1.0)]

    assert normalized_polygon_pixels(polygon, 200, 100) == [
        (20, 20),
        (150, 20),
        (150, 100),
        (20, 100),
    ]


def test_overlay_uses_copy_and_distinguishes_saved_and_draft_zones():
    frame = np.full((100, 200, 3), 240, dtype=np.uint8)
    original = frame.copy()
    saved = ZoneRecord(
        id="saved",
        display_name="Saved",
        polygon=[(0.05, 0.1), (0.35, 0.1), (0.35, 0.9), (0.05, 0.9)],
    )
    draft = ZoneRecord(
        id="draft",
        display_name="Draft",
        polygon=[(0.55, 0.1), (0.95, 0.1), (0.95, 0.9), (0.55, 0.9)],
    )

    rendered = render_zone_overlay(frame, [saved], draft_zone=draft)

    assert np.array_equal(frame, original)
    assert not np.array_equal(rendered[50, 10], original[50, 10])
    assert not np.array_equal(rendered[50, 110], original[50, 110])
    assert not np.array_equal(rendered[50, 10], rendered[50, 110])


def test_preview_empty_state_and_preview_do_not_write():
    facade = MockFacade()
    initial = [zone.model_dump() for zone in facade.get_zones()]

    preview, status = callbacks.preview_zone_changes(
        facade, None, "Shelf", "shelf", 0.1, 0.2, 0.5, 0.8
    )
    assert preview is None
    assert "Capture a calibration image first" in status

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    preview, status = callbacks.preview_zone_changes(
        facade, frame, "Shelf", "shelf", 0.1, 0.2, 0.5, 0.8
    )
    assert preview.shape == frame.shape
    assert "Nothing has been saved" in status
    assert [zone.model_dump() for zone in facade.get_zones()] == initial


def test_capture_calibration_uses_one_explicit_snapshot():
    class CapturingFacade(MockFacade):
        capture_calls = 0

        def capture_camera_frame(self):
            self.capture_calls += 1
            return np.full((80, 120, 3), 220, dtype=np.uint8)

    facade = CapturingFacade()

    base, preview, status = callbacks.capture_calibration_image(facade)

    assert facade.capture_calls == 1
    assert np.all(base == 220)
    assert not np.array_equal(preview, base)
    assert "frozen base image is unchanged" in status


def test_existing_zone_update_allowed_at_four_zone_limit(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        database_path=str(tmp_path / "data" / "memory.db"),
        gradio_temp_dir=str(tmp_path / "tmp"),
        inference_mode="mock",
    )
    facade = LocalRecallFacadeImpl(settings, mock=True)
    try:
        facade.load_demo_zones()
        facade.add_zone("Shelf", "shelf", 0.05, 0.05, 0.25, 0.25)
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        facade.enroll_object("Keys", "", image)
        event = facade.analyze_pair(image, image)
        assert event.to_zone_id == "shelf"

        updated = facade.update_zone("Main Shelf", "shelf", 0.1, 0.1, 0.4, 0.4)

        assert updated.id == "shelf"
        assert updated.display_name == "Main Shelf"
        assert facade.store.get_event(event.id).to_zone_id == "shelf"
        assert len(facade.get_zones()) == 4
        with pytest.raises(ValueError, match="limited to four"):
            facade.add_zone("Drawer", "drawer", 0.6, 0.1, 0.9, 0.4)
    finally:
        facade.close()


def test_invalid_or_changed_existing_id_performs_no_write():
    facade = MockFacade()
    before = [zone.model_dump() for zone in facade.get_zones()]

    with pytest.raises(ValueError, match="Zone bounds must satisfy"):
        callbacks.save_zone_changes(
            facade, "left_tray", "Left Tray", "left_tray", 0.8, 0.1, 0.2, 0.9
        )
    with pytest.raises(ValueError, match="fixed"):
        callbacks.save_zone_changes(
            facade, "left_tray", "Left Tray", "renamed_id", 0.1, 0.1, 0.3, 0.9
        )

    assert [zone.model_dump() for zone in facade.get_zones()] == before


def test_save_zone_changes_creates_and_updates_expected_records():
    facade = MockFacade()

    created = callbacks.save_zone_changes(
        facade,
        callbacks.NEW_ZONE_VALUE,
        "Shelf",
        "shelf",
        0.05,
        0.1,
        0.25,
        0.9,
    )
    assert "Created zone" in created

    updated = callbacks.save_zone_changes(
        facade, "shelf", "Upper Shelf", "shelf", 0.1, 0.2, 0.3, 0.8
    )
    shelf = next(zone for zone in facade.get_zones() if zone.id == "shelf")
    assert "Updated zone" in updated
    assert shelf.display_name == "Upper Shelf"
    assert shelf.polygon == [(0.1, 0.2), (0.3, 0.2), (0.3, 0.8), (0.1, 0.8)]
