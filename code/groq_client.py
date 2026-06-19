from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from utils import ImagePayload


DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class GroqUnavailable(RuntimeError):
    pass


@dataclass
class GroqResult:
    parsed: dict[str, Any]
    raw_text: str
    model: str
    attempts: int


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Groq response JSON was not an object")
    return value


class GroqClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> None:
        if not self.api_key:
            raise GroqUnavailable("GROQ_API_KEY is not set")
        if self._client is not None:
            return
        try:
            from groq import Groq
        except Exception as exc:
            raise GroqUnavailable("Groq SDK is not installed. Install with: pip install groq") from exc
        self._client = Groq(api_key=self.api_key)

    def generate_json(self, prompt: str, images: list[ImagePayload]) -> GroqResult:
        self._ensure_client()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = self._call_model(prompt, images)
                parsed = parse_json_object(raw_text)
                return GroqResult(parsed=parsed, raw_text=raw_text, model=self.model, attempts=attempt)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = self.retry_base_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(sleep_seconds)
        raise GroqUnavailable(f"Groq call failed after {self.max_retries} attempts: {last_error}")

    def _call_model(self, prompt: str, images: list[ImagePayload]) -> str:
        assert self._client is not None
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images[:5]:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"},
                }
            )

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = completion.choices[0].message.content
        if not text:
            raise ValueError("Groq response did not contain message content")
        return text
