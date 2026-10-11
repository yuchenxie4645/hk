import numpy as np
import pytest

from app import camera
from app.camera import SparkCameraSource
from app.config import Settings
from app.facade import LocalRecallFacadeImpl
from app.ui.app import create_app
from app.ui.mock_facade import MockFacade


class FakeCapture:
    instances: list["FakeCapture"] = []

    def __init__(self, frames, *, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self):  # noqa: N802 - mirrors cv2.VideoCapture
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return frame is not None, frame

    def release(self):
        self.released = True


def install_fake(monkeypatch, frames, *, opened=True):
    FakeCapture.instances = []
    calls = []

    def factory(index, api):
        calls.append((index, api))
        return FakeCapture(frames, opened=opened)

    monkeypatch.setattr(camera.cv2, "VideoCapture", factory)
    return calls


def bgr(value_b, value_g, value_r, shape=(4, 6)):
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = value_b, value_g, value_r
    return frame


def test_capture_returns_last_good_warmup_frame_as_rgb_and_releases(monkeypatch):
    calls = install_fake(monkeypatch, [bgr(1, 2, 3), bgr(10, 20, 30), None])
    frame = SparkCameraSource(3, warmup_reads=3).capture()
    assert calls == [(3, camera.cv2.CAP_V4L2)]
    assert frame.dtype == np.uint8 and frame.shape == (4, 6, 3)
    assert frame.flags["C_CONTIGUOUS"]
    assert tuple(frame[0, 0]) == (30, 20, 10)
    assert FakeCapture.instances[0].released


def test_unopened_device_raises_actionable_error_and_releases(monkeypatch):
    install_fake(monkeypatch, [], opened=False)
    with pytest.raises(RuntimeError, match="Spark camera 0 could not be opened.*SPARK_CAMERA_INDEX"):
        SparkCameraSource(0).capture()
    assert FakeCapture.instances[0].released


def test_no_frames_raises_and_releases(monkeypatch):
    install_fake(monkeypatch, [None, np.zeros((4, 4), dtype=np.uint8)])
    with pytest.raises(RuntimeError, match="returned no frames; upload images instead"):
        SparkCameraSource(0, warmup_reads=4).capture()
    assert FakeCapture.instances[0].released


def test_oversized_frame_is_downscaled_within_facade_limit(monkeypatch):
    install_fake(monkeypatch, [np.zeros((3000, 4000, 3), dtype=np.uint8)])
    frame = SparkCameraSource(0, warmup_reads=1).capture()
    height, width = frame.shape[:2]
    assert width <= 1920 and height <= 1080
    assert (width, height) == (1440, 1080)


def test_busy_camera_raises_instead_of_opening_a_second_handle(monkeypatch):
    calls = install_fake(monkeypatch, [bgr(1, 2, 3)])
    source = SparkCameraSource(0, lock_timeout_seconds=0.01)
    source._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            source.capture()
    finally:
        source._lock.release()
    assert calls == []


def test_backend_facade_uses_configured_index_and_reuses_source(monkeypatch, tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        database_path=str(tmp_path / "data" / "memory.db"),
        gradio_temp_dir=str(tmp_path / "data" / "tmp"),
        inference_mode="mock",
        spark_camera_index=2,
    )
    facade = LocalRecallFacadeImpl(settings, mock=True)
    try:
        calls = install_fake(monkeypatch, [bgr(1, 2, 3)])
        assert facade.capture_camera_frame().shape == (4, 6, 3)
        first_source = facade._spark_camera
        install_fake(monkeypatch, [bgr(1, 2, 3)])
        facade.capture_camera_frame()
        assert facade._spark_camera is first_source
        assert calls[0][0] == 2
        assert facade.store.count_events() == 0
    finally:
        facade.close()


def test_mock_facade_never_constructs_video_capture(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("mock mode must not open a camera")

    monkeypatch.setattr(camera.cv2, "VideoCapture", forbidden)
    with pytest.raises(RuntimeError, match="MOCK mode does not use the Spark camera"):
        MockFacade().capture_camera_frame()


def test_mock_ui_exposes_spark_capture_buttons():
    config_text = str(create_app(MockFacade()).get_config_file())
    assert "Capture BEFORE (Spark camera)" in config_text
    assert "Capture AFTER (Spark camera)" in config_text
