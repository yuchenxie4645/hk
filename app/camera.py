"""Frame sources: browser-pushed frames, fixture videos, and explicit Spark camera snapshots.

`SparkCameraSource` is the only code allowed to open a server-local camera device. It opens the
device per user-triggered snapshot, releases it immediately, never writes video, and never calls
the VLM.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSample:
    frame_rgb: np.ndarray
    received_monotonic: float
    source_id: str


class BrowserPushSource:
    """Single-producer, bounded RAM ring for frames pushed by the laptop browser."""

    def __init__(
        self,
        *,
        max_width: int = 1280,
        stale_seconds: float = 4.0,
        max_frames: int = 60,
        max_pixels: int = 1920 * 1080,
    ):
        self.max_width = max_width
        self.stale_seconds = stale_seconds
        self.max_pixels = max_pixels
        self._ring: deque[FrameSample] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._source_id: str | None = None
        self._frames_received = 0
        self._dropped_frames = 0
        self._source_disconnects = 0

    def ingest_frame(self, frame: np.ndarray, source_id: str) -> dict:
        if not source_id or not source_id.strip():
            raise ValueError("source_id is required")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("browser frames must be uint8 NumPy arrays")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("browser frames must be nonempty RGB HxWx3 arrays")
        height, width = frame.shape[:2]
        if height * width > self.max_pixels:
            raise ValueError("browser frame exceeds the decoded pixel limit")
        if width > 3840 or height > 2160:
            raise ValueError("browser frame exceeds the maximum supported geometry")
        if width > self.max_width:
            scaled_height = max(1, round(height * self.max_width / width))
            frame = cv2.resize(frame, (self.max_width, scaled_height), interpolation=cv2.INTER_AREA)
        copy = np.ascontiguousarray(frame.copy())
        received = time.monotonic()
        with self._lock:
            if self._source_id and self._source_id != source_id:
                if self._ring and received - self._ring[-1].received_monotonic <= self.stale_seconds:
                    self._dropped_frames += 1
                    return {
                        "accepted": False,
                        "reason": "another browser camera currently owns the source lease",
                        **self._health_locked(received),
                    }
                self._source_disconnects += 1
                self._ring.clear()
            self._source_id = source_id
            if len(self._ring) == self._ring.maxlen:
                self._dropped_frames += 1
            self._ring.append(FrameSample(copy, received, source_id))
            self._frames_received += 1
            return {"accepted": True, **self._health_locked(received)}

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if not self._ring:
                return None
            return self._ring[-1].frame_rgb.copy()

    def samples(self) -> list[FrameSample]:
        with self._lock:
            return [FrameSample(item.frame_rgb.copy(), item.received_monotonic, item.source_id) for item in self._ring]

    def health(self) -> dict:
        with self._lock:
            return self._health_locked(time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._source_id = None

    def _health_locked(self, now: float) -> dict:
        last = self._ring[-1].received_monotonic if self._ring else None
        age = None if last is None else max(0.0, now - last)
        connected = age is not None and age <= self.stale_seconds
        samples = list(self._ring)[-10:]
        if len(samples) > 1:
            elapsed = samples[-1].received_monotonic - samples[0].received_monotonic
            fps = (len(samples) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0
        return {
            "connected": connected,
            "source_id": self._source_id if connected else None,
            "seconds_since_last_frame": age,
            "frames_received": self._frames_received,
            "dropped_frames": self._dropped_frames,
            "source_disconnects": self._source_disconnects,
            "ingest_fps": round(fps, 2),
            "buffered_frames": len(self._ring),
        }


def _is_valid_bgr(frame: object) -> bool:
    return (
        isinstance(frame, np.ndarray)
        and frame.dtype == np.uint8
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
    )


class SparkCameraSource:
    """Single still snapshots from a camera attached to the Spark, opened only for the capture."""

    def __init__(
        self,
        index: int = 0,
        *,
        warmup_reads: int = 8,
        lock_timeout_seconds: float = 2.0,
        max_width: int = 1920,
        max_height: int = 1080,
    ):
        self.index = index
        self.warmup_reads = max(1, warmup_reads)
        self.lock_timeout_seconds = lock_timeout_seconds
        self.max_width = max_width
        self.max_height = max_height
        self._lock = threading.Lock()

    def capture(self) -> np.ndarray:
        if not self._lock.acquire(timeout=self.lock_timeout_seconds):
            raise RuntimeError(
                f"Spark camera {self.index} is busy with another capture; try again in a moment."
            )
        try:
            return self._capture_locked()
        finally:
            self._lock.release()

    def _capture_locked(self) -> np.ndarray:
        frame: np.ndarray | None = None
        capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        try:
            if not capture.isOpened():
                raise RuntimeError(
                    f"Spark camera {self.index} could not be opened; check the cable or "
                    "SPARK_CAMERA_INDEX, or upload images instead."
                )
            for _ in range(self.warmup_reads):
                ok, bgr = capture.read()
                if ok and _is_valid_bgr(bgr):
                    frame = bgr
        finally:
            capture.release()
        if frame is None:
            raise RuntimeError(f"Spark camera {self.index} returned no frames; upload images instead.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(self.max_width / width, self.max_height / height)
        if scale < 1.0:
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb)


class FixtureSource:
    """Decode a user-supplied video file; never opens a camera device index."""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)

    def frames(self) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open fixture video: {self.video_path}")
        result = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                result.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        return result

