"""Stable in-process facade that coordinates local VLM, SQLite, and evidence."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from PIL import Image

from app.camera import BrowserPushSource, SparkCameraSource
from app.composite import build_composite
from app.config import Settings, get_settings
from app.event_analyzer import EventAnalyzer
from app.memory_store import MemoryStore
from app.privacy import PrivacyJanitor, validate_transient_dir
from app.query_service import QueryService
from app.schemas import (
    EventContext,
    EventRecord,
    ObjectRecord,
    QueryResult,
    SystemStatus,
    VLMEventResult,
    ZoneRecord,
)
from app.vlm_client import MockVLMClient, OpenAICompatibleVLMClient, VLMClient


def normalize_object_id(display_name: str) -> str:
    object_id = re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")
    if not object_id or not object_id[0].isalpha():
        raise ValueError("Object name must produce a lowercase ID beginning with a letter")
    return object_id


def parse_object_aliases(description: str) -> list[str]:
    return sorted(
        {
            alias.strip()
            for alias in re.split(r"[,;]", description or "")
            if alias.strip()
        }
    )


def build_zone_record(
    display_name: str,
    zone_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    active_zone_count: int,
) -> ZoneRecord:
    display_name = display_name.strip()
    zone_id = zone_id.strip().lower()
    if not display_name or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", zone_id):
        raise ValueError("Enter a zone name and a lowercase ID such as `shelf`.")
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Zone bounds must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.")
    if zone_id == "outside_view":
        raise ValueError("outside_view is reserved for system events.")
    if active_zone_count >= 4:
        raise ValueError("The workspace is limited to four named zones.")
    return ZoneRecord(
        id=zone_id,
        display_name=display_name,
        polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
    )


DEFAULT_ZONES = (
    ZoneRecord(
        id="left_tray",
        display_name="Left Tray",
        polygon=[(0.02, 0.10), (0.32, 0.10), (0.32, 0.92), (0.02, 0.92)],
    ),
    ZoneRecord(
        id="center_mat",
        display_name="Center Mat",
        polygon=[(0.35, 0.10), (0.65, 0.10), (0.65, 0.92), (0.35, 0.92)],
    ),
    ZoneRecord(
        id="right_tray",
        display_name="Right Tray",
        polygon=[(0.68, 0.10), (0.98, 0.10), (0.98, 0.92), (0.68, 0.92)],
    ),
)


class LocalRecallFacadeImpl:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        mock: bool | None = None,
        vlm_client: VLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.mock = self.settings.inference_mode == "mock" if mock is None else mock
        self.data_dir = Path(self.settings.data_dir).resolve()
        self.events_dir = self.data_dir / "events"
        self.objects_dir = self.data_dir / "objects"
        self.temp_dir = validate_transient_dir(self.settings.gradio_temp_dir, self.data_dir)
        for directory in (self.data_dir, self.events_dir, self.objects_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.privacy = PrivacyJanitor(self.temp_dir)
        self.privacy.start()
        db_path = Path(self.settings.database_path)
        if not db_path.is_absolute():
            db_path = (Path.cwd() / db_path).resolve()
        self.store = MemoryStore(db_path)
        self.vlm = vlm_client or (
            MockVLMClient() if self.mock else OpenAICompatibleVLMClient(self.settings)
        )
        self.analyzer = EventAnalyzer(self.vlm)
        self.queries = QueryService(self.store, None if self.mock else self.vlm)
        self.frame_source = BrowserPushSource(
            max_width=self.settings.browser_max_width,
            stale_seconds=self.settings.browser_source_stale_seconds,
        )
        self._spark_camera: SparkCameraSource | None = None
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="localrecall-vlm")
        self._inference_slot = threading.BoundedSemaphore(value=1)
        self._state_lock = threading.Lock()
        self._watch_mode = False
        self._last_error: str | None = None

    def get_status(self) -> SystemStatus:
        source = self.frame_source.health()
        model_ready = False
        if not self.mock:
            try:
                health_url = self.settings.vlm_base_url.rstrip("/")
                health_url = health_url.removesuffix("/v1")
                response = httpx.get(
                    f"{health_url}/health",
                    timeout=0.4,
                    follow_redirects=False,
                )
                model_ready = response.status_code == 200
            except httpx.HTTPError:
                model_ready = False
        return SystemStatus(
            app_mode="backend",
            offline_mode=self.settings.offline_mode,
            model_name=self.vlm.model_name,
            model_ready=model_ready,
            model_endpoint=self.settings.vlm_base_url,
            source_connected=source["connected"],
            source_id=source["source_id"],
            seconds_since_last_frame=source["seconds_since_last_frame"],
            frames_received=source["frames_received"],
            dropped_frames=source["dropped_frames"],
            watch_mode=self._watch_mode,
            event_count=self.store.count_events(),
            evidence_count=self.store.count_evidence(),
            temp_cache_files=self.privacy.stats()["files"],
            temp_cache_bytes=self.privacy.stats()["bytes"],
            last_error=self._last_error,
        )

    def get_latest_frame(self) -> np.ndarray | None:
        return self.frame_source.latest()

    def ingest_frame(self, frame: np.ndarray, source_id: str) -> dict:
        receipt = self.frame_source.ingest_frame(frame, source_id)
        return receipt

    def capture_camera_frame(self) -> np.ndarray:
        with self._state_lock:
            if self._spark_camera is None:
                self._spark_camera = SparkCameraSource(self.settings.spark_camera_index)
            camera = self._spark_camera
        return camera.capture()

    def analyze_pair(self, before: np.ndarray, after: np.ndarray) -> EventRecord:
        before = self._validate_frame(before, "BEFORE")
        after = self._validate_frame(after, "AFTER")
        if not self._inference_slot.acquire(blocking=False):
            raise RuntimeError("An event is already being analyzed; wait for it to finish.")
        future = self._inference_executor.submit(self._analyze_pair_worker, before, after)
        future.add_done_callback(lambda _: self._inference_slot.release())
        return future.result()

    @staticmethod
    def _validate_frame(frame: np.ndarray, label: str) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError(f"{label} image must be a uint8 RGB NumPy array")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"{label} image must be nonempty HxWx3 RGB")
        height, width = frame.shape[:2]
        if width > 1920 or height > 1080 or width * height > 1920 * 1080:
            raise ValueError(f"{label} image exceeds the 1920x1080 input limit")
        return np.ascontiguousarray(frame.copy())

    def _analyze_pair_worker(self, before: np.ndarray, after: np.ndarray) -> EventRecord:
        zones = self.store.list_zones()
        objects = self.store.list_objects()
        if len(zones) < 2:
            raise ValueError("Load demo zones before analyzing an image pair.")
        if not objects:
            raise ValueError("Enroll at least one object before analyzing an image pair.")
        event_key = uuid4().hex
        working = self.temp_dir / f"{event_key}_composite.jpg"
        before_time = datetime.now(UTC)
        after_time = datetime.now(UTC)
        references = self._load_references(objects)
        composite_path = build_composite(
            before,
            after,
            zones,
            objects,
            working,
            reference_images=references,
            before_time=before_time,
            after_time=after_time,
        )
        context = EventContext(objects=objects, zones=zones)
        try:
            result, latency_ms, raw = self.analyzer.analyze(composite_path, context)
        except Exception:
            working.unlink(missing_ok=True)
            raise
        accepted = result.confidence >= self.settings.auto_accept_confidence and result.event_type != "uncertain"
        reviewable = result.confidence >= self.settings.review_confidence and not accepted
        if result.confidence < self.settings.review_confidence:
            result = result.model_copy(
                update={
                    "event_type": "uncertain",
                    "uncertainty_reason": result.uncertainty_reason or "confidence below review threshold",
                }
            )

        pre_path: Path | None = None
        post_path: Path | None = None
        saved_composite: Path | None = None
        if accepted or reviewable:
            evidence_dir = self.events_dir / event_key
            evidence_dir.mkdir(parents=True, exist_ok=False)
            pre_path = evidence_dir / "before.jpg"
            post_path = evidence_dir / "after.jpg"
            saved_composite = evidence_dir / "composite.jpg"
            self._save_rgb(before, pre_path)
            self._save_rgb(after, post_path)
            shutil.copy2(composite_path, saved_composite)
        try:
            record = self.store.add_event(
                result,
                accepted=accepted,
                model_name=self.vlm.model_name,
                model_latency_ms=latency_ms,
                raw_response_json=raw,
                pre_image_path=str(pre_path) if pre_path else None,
                post_image_path=str(post_path) if post_path else None,
                composite_path=str(saved_composite) if saved_composite else None,
            )
        except Exception:
            if pre_path:
                shutil.rmtree(pre_path.parent, ignore_errors=True)
            raise
        finally:
            working.unlink(missing_ok=True)
        with self._state_lock:
            self._last_error = None
        return record

    @staticmethod
    def _save_rgb(frame: np.ndarray, path: Path) -> None:
        Image.fromarray(frame).save(path, format="JPEG", quality=90, optimize=True)

    @staticmethod
    def _load_references(objects: list[ObjectRecord]) -> dict[str, np.ndarray]:
        references: dict[str, np.ndarray] = {}
        for obj in objects:
            if not obj.reference_image_path:
                continue
            path = Path(obj.reference_image_path)
            if path.exists():
                with Image.open(path) as image:
                    references[obj.id] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return references

    def get_objects(self) -> list[ObjectRecord]:
        return self.store.list_objects()

    def get_zones(self) -> list[ZoneRecord]:
        return self.store.list_zones()

    def get_events(self, limit: int = 50) -> list[EventRecord]:
        return self.store.list_events(limit)

    def get_current_state(self) -> list[dict]:
        return self.store.get_state_rows()

    def delete_object(self, object_id: str) -> None:
        reference_path = self.store.delete_object(object_id)
        if reference_path:
            Path(reference_path).unlink(missing_ok=True)

    def delete_zone(self, zone_id: str) -> None:
        if zone_id == "outside_view":
            raise ValueError("outside_view is reserved for system events.")
        self.store.delete_zone(zone_id)

    def start_watch(self) -> SystemStatus:
        raise RuntimeError(
            "Automatic motion analysis is not enabled yet. The browser stream is ingress-only; use the manual image-pair flow."
        )

    def stop_watch(self) -> SystemStatus:
        with self._state_lock:
            self._watch_mode = False
        return self.get_status()

    def ask(self, question: str) -> QueryResult:
        return self.queries.ask(question)

    def enroll_object(
        self, display_name: str, description: str, reference_image: np.ndarray
    ) -> ObjectRecord:
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Object name is required")
        if len(self.store.list_objects()) >= 5:
            raise ValueError("The demo object registry is limited to five active objects")
        image = self._validate_frame(reference_image, "Reference")
        object_id = normalize_object_id(display_name)
        if any(obj.id == object_id for obj in self.store.list_objects(active_only=False)):
            raise ValueError(f"Object ID already exists: {object_id}")
        aliases = parse_object_aliases(description)
        object_dir = self.objects_dir / object_id
        object_dir.mkdir(parents=True, exist_ok=False)
        reference_path = object_dir / "reference.jpg"
        pil = Image.fromarray(image)
        side = min(pil.width, pil.height)
        left, top = (pil.width - side) // 2, (pil.height - side) // 2
        pil.crop((left, top, left + side, top + side)).resize((256, 256)).save(
            reference_path, format="JPEG", quality=90
        )
        record = ObjectRecord(
            id=object_id,
            display_name=display_name,
            aliases=aliases,
            reference_image_path=str(reference_path),
        )
        try:
            return self.store.add_object(record)
        except Exception:
            shutil.rmtree(object_dir, ignore_errors=True)
            raise

    def load_demo_zones(self) -> list[ZoneRecord]:
        for zone in DEFAULT_ZONES:
            self.store.upsert_zone(zone)
        return self.store.list_zones()

    def add_zone(
        self, display_name: str, zone_id: str, x1: float, y1: float, x2: float, y2: float
    ) -> ZoneRecord:
        record = build_zone_record(
            display_name,
            zone_id,
            x1,
            y1,
            x2,
            y2,
            active_zone_count=len(self.store.list_zones()),
        )
        return self.store.add_zone(record)

    def approve_event(self, event_id: int) -> EventRecord:
        return self.store.set_event_decision(event_id, True)

    def reject_event(self, event_id: int) -> EventRecord:
        return self.store.set_event_decision(event_id, False)

    def correct_event(
        self,
        event_id: int,
        event_type: str,
        object_id: str | None,
        from_zone_id: str | None,
        to_zone_id: str | None,
    ) -> EventRecord:
        existing = self.store.get_event(event_id)
        if event_type not in {"move", "appear", "disappear", "no_change"}:
            raise ValueError("Choose move, appear, disappear, or no_change for an accepted correction.")
        candidate = VLMEventResult(
            event_type=event_type,
            object_id=object_id,
            from_zone_id=from_zone_id,
            to_zone_id=to_zone_id,
            confidence=existing.confidence,
            visible_evidence=existing.visible_evidence,
            uncertainty_reason=None,
        )
        from app.event_analyzer import validate_event

        validate_event(candidate.model_dump(), EventContext(objects=self.get_objects(), zones=self.get_zones()))
        return self.store.correct_event(
            event_id,
            event_type=event_type,
            object_id=object_id,
            from_zone_id=from_zone_id,
            to_zone_id=to_zone_id,
        )

    def get_health(self) -> dict:
        status = self.get_status()
        database_status = self.store.integrity_check()
        return {
            "ok": database_status == "ok",
            "database": database_status,
            "model_ready": status.model_ready,
            "model_name": status.model_name,
            "offline_mode": status.offline_mode,
            "source": self.frame_source.health(),
            "event_count": status.event_count,
            "evidence_count": status.evidence_count,
            "transient_cache": self.privacy.stats(),
        }

    def export_session(self) -> Path:
        exports_dir = self.data_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now(UTC).strftime("session-%Y%m%dT%H%M%SZ.zip")
        output = exports_dir / name
        snapshot = self.temp_dir / f"{uuid4().hex}.sqlite3"
        with self.store.connect() as source, sqlite3.connect(snapshot) as destination:
            source.backup(destination)
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(snapshot, "localrecall.db")
                for directory in (self.events_dir, self.objects_dir):
                    if directory.exists():
                        for path in directory.rglob("*"):
                            if path.is_file():
                                archive.write(path, path.relative_to(self.data_dir))
                archive.writestr("metadata.json", json.dumps({"model": self.vlm.model_name, "created_at": datetime.now(UTC).isoformat()}, indent=2))
        finally:
            snapshot.unlink(missing_ok=True)
        return output

    def delete_all_memory(self) -> None:
        self.store.delete_all_memory()
        self.store.initialize()
        for directory in (self.events_dir, self.objects_dir):
            if directory.exists():
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
        self.privacy.clear()
        self.frame_source.clear()
        with self._state_lock:
            self._watch_mode = False
            self._last_error = None

    def close(self) -> None:
        self._inference_executor.shutdown(wait=True, cancel_futures=True)
        self.privacy.stop(clear=True)

