"""Local OpenAI-compatible VLM client and deterministic development mock."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Protocol

import httpx

from app.config import Settings, is_loopback_url
from app.schemas import EventContext


class VLMClient(Protocol):
    model_name: str

    def analyze_event(
        self, composite_path: Path, context: EventContext
    ) -> tuple[str, int, dict]: ...

    def repair_event(
        self, composite_path: Path, context: EventContext, response: str, error: str
    ) -> tuple[str, int, dict]: ...

    def answer_from_records(self, question: str, records: list[dict]) -> str: ...


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _read_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent / "prompts" / name
    return path.read_text(encoding="utf-8")


def render_event_prompt(context: EventContext) -> str:
    prompt = _read_prompt("event_extractor.txt")
    return prompt.replace(
        "{{OBJECT_IDS}}", ", ".join(obj.id for obj in context.objects) or "none"
    ).replace("{{ZONE_IDS}}", ", ".join(zone.id for zone in context.zones) or "none")


class OpenAICompatibleVLMClient:
    _citation_pattern = re.compile(r"#(\d+)\b")
    _unsupported_answer = "No reliable observation is available in the saved events."

    def __init__(self, settings: Settings):
        if settings.offline_mode and not is_loopback_url(settings.vlm_base_url):
            raise ValueError("Offline model access must use a loopback URL")
        self.base_url = settings.vlm_base_url.rstrip("/")
        self.model_name = settings.vlm_model
        self.timeout = settings.vlm_timeout_seconds

    def _post(self, payload: dict) -> tuple[str, int, dict]:
        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("VLM returned an empty or invalid message content")
        # Store response JSON but never request headers or credentials.
        return content.strip(), elapsed_ms, body

    def analyze_event(self, composite_path: Path, context: EventContext) -> tuple[str, int, dict]:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": render_event_prompt(context)},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(composite_path)},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return self._post(payload)

    def repair_event(
        self, composite_path: Path, context: EventContext, response: str, error: str
    ) -> tuple[str, int, dict]:
        repair_prompt = _read_prompt("event_repair.txt").replace("{{ERROR}}", error[:1000]).replace(
            "{{RESPONSE}}", response[:3000]
        )
        prompt = render_event_prompt(context) + "\n\n" + repair_prompt
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(composite_path)},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return self._post(payload)

    def answer_from_records(self, question: str, records: list[dict]) -> str:
        allowed_event_ids: set[int] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            raw_id = record.get("id")
            if (
                isinstance(raw_id, (int, str))
                and not isinstance(raw_id, bool)
                and str(raw_id).isdigit()
            ):
                allowed_event_ids.add(int(raw_id))
        if not allowed_event_ids:
            return self._unsupported_answer

        prompt = _read_prompt("query_answer.txt").replace("{{QUESTION}}", question).replace(
            "{{RECORDS}}", json.dumps(records, ensure_ascii=False)[:12000]
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        content, _, _ = self._post(payload)
        cited_ids = {int(match) for match in self._citation_pattern.findall(content)}
        if not cited_ids or not cited_ids.issubset(allowed_event_ids):
            return self._unsupported_answer
        return content


class MockVLMClient:
    """Deterministic fixture client. Its output is always marked with model_name=mock."""

    model_name = "mock/localrecall-deterministic"

    def analyze_event(self, composite_path: Path, context: EventContext) -> tuple[str, int, dict]:
        if not context.objects or len(context.zones) < 2:
            result = {
                "event_type": "uncertain",
                "object_id": None,
                "from_zone_id": None,
                "to_zone_id": None,
                "confidence": 0.0,
                "visible_evidence": [],
                "uncertainty_reason": "mock needs one registered object and at least two zones",
            }
        else:
            result = {
                "event_type": "move",
                "object_id": context.objects[0].id,
                "from_zone_id": context.zones[0].id,
                "to_zone_id": context.zones[-1].id,
                "confidence": 0.91,
                "visible_evidence": [
                    "Mock fixture reports the registered object in the source zone before.",
                    "Mock fixture reports the registered object in the destination zone after.",
                ],
                "uncertainty_reason": None,
            }
        raw = json.dumps(result)
        return raw, 1, {"mock": True, "choices": [{"message": {"content": raw}}]}

    def repair_event(
        self, composite_path: Path, context: EventContext, response: str, error: str
    ) -> tuple[str, int, dict]:
        return self.analyze_event(composite_path, context)

    def answer_from_records(self, question: str, records: list[dict]) -> str:
        return "MOCK: no deterministic answer is defined for this question."

