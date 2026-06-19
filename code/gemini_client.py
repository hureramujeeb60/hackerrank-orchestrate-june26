from __future__ import annotations

import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from schema import default_prediction, parse_json_object


class GeminiClient:
    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path = ".cache/gemini_claims",
        max_retries: int = 2,
        retry_sleep: float = 2.0,
        use_cache: bool = True,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.use_cache = use_cache
        self._client = None
        self._types = None
        self._sdk_error = None
        self._init_sdk()

    @property
    def available(self) -> bool:
        return bool(self.api_key and self._client and self._types)

    def _init_sdk(self) -> None:
        if not self.api_key:
            self._sdk_error = "GEMINI_API_KEY or GOOGLE_API_KEY is not set"
            return
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            self._client = genai.Client(api_key=self.api_key)
            self._types = types
        except Exception as exc:  # pragma: no cover - depends on local environment
            self._sdk_error = f"Google Gen AI SDK unavailable: {exc}"

    def generate_json(
        self,
        prompt: str,
        images: List[Dict[str, object]],
        cache_key: str,
        fallback_reason: str,
    ) -> Dict[str, object]:
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        if not self.available:
            result = default_prediction(f"{fallback_reason} ({self._sdk_error}).", [str(i["id"]) for i in images])
            self._write_cache(cache_key, result, source="fallback")
            return result

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw_text = self._call_gemini(prompt, images)
                parsed = parse_json_object(raw_text)
                self._write_cache(cache_key, parsed, source="gemini")
                return parsed
            except Exception as exc:  # pragma: no cover - live API path
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (attempt + 1))

        result = default_prediction(self._safe_fallback_reason(fallback_reason, last_error), [str(i["id"]) for i in images])
        self._write_cache(cache_key, result, source="fallback")
        return result

    def _safe_fallback_reason(self, fallback_reason: str, error: str) -> str:
        text = error.lower()
        if "429" in text or "rate limit" in text or "quota" in text:
            return f"{fallback_reason} Provider rate limit or quota was reached."
        if "401" in text or "unauthorized" in text or "api key" in text:
            return f"{fallback_reason} Provider authentication failed."
        if "413" in text or "request too large" in text:
            return f"{fallback_reason} Provider rejected the image request as too large."
        return f"{fallback_reason} Provider response was unavailable or invalid."

    def _call_gemini(self, prompt: str, images: List[Dict[str, object]]) -> str:
        assert self._client is not None
        assert self._types is not None
        parts = []
        for image in images:
            path = Path(str(image["path"]))
            parts.append(
                self._types.Part.from_bytes(
                    data=self._read_image_bytes(path),
                    mime_type=str(image.get("mime_type") or "image/jpeg"),
                )
            )
        if hasattr(self._types.Part, "from_text"):
            parts.append(self._types.Part.from_text(text=prompt))
        else:
            parts.append(self._types.Part(text=prompt))
        contents = [self._types.Content(role="user", parts=parts)]

        config = None
        try:
            config = self._types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            )
        except TypeError:
            config = self._types.GenerateContentConfig(temperature=0)

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        text = getattr(response, "text", None)
        if text:
            return text
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
            return "\n".join(str(getattr(part, "text", "") or "") for part in parts)
        return ""

    def _read_image_bytes(self, path: Path, max_bytes: int = 7_000_000) -> bytes:
        data = path.read_bytes()
        if len(data) <= max_bytes:
            return data
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as img:
                img.thumbnail((1800, 1800))
                rgb = img.convert("RGB")
                buffer = BytesIO()
                rgb.save(buffer, format="JPEG", quality=88, optimize=True)
                compressed = buffer.getvalue()
                return compressed if compressed else data
        except Exception:
            return data

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> Dict[str, object] | None:
        if not self.use_cache:
            return None
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("response") if isinstance(payload, dict) else None
        except Exception:
            return None

    def _write_cache(self, cache_key: str, response: Dict[str, object], source: str) -> None:
        if not self.use_cache:
            return
        payload = {
            "model": self.model,
            "source": source,
            "response": response,
        }
        with self._cache_path(cache_key).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
