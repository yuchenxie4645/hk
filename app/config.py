"""Environment-backed settings and local-only endpoint validation."""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def is_loopback_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        return False
    hostname = hostname.lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 7860
    data_dir: str = "./data"
    database_path: str = "./data/localrecall.db"
    vlm_base_url: str = "http://127.0.0.1:8000/v1"
    vlm_model: str = "nvidia/Qwen3.8-27B-NVFP4"
    vlm_timeout_seconds: float = 180.0
    offline_mode: bool = True
    inference_mode: str = "mock"
    frame_source: str = "browser"
    browser_stream_interval_sec: float = 0.5
    browser_max_width: int = 1280
    browser_source_stale_seconds: float = 4.0
    spark_camera_index: int = 0
    auto_accept_confidence: float = 0.70
    review_confidence: float = 0.45
    gradio_temp_dir: str = "./data/tmp/gradio"

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> Settings:
        if self.offline_mode and not is_loopback_url(self.vlm_base_url):
            raise ValueError("OFFLINE_MODE=1 requires VLM_BASE_URL to use a loopback host")
        if not 0 <= self.review_confidence <= self.auto_accept_confidence <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= review <= auto_accept <= 1")
        if self.inference_mode not in {"mock", "real"}:
            raise ValueError("INFERENCE_MODE must be mock or real")
        if self.browser_max_width < 64:
            raise ValueError("BROWSER_MAX_WIDTH must be at least 64 pixels")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

