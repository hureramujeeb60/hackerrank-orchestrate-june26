from __future__ import annotations

import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from schema import default_prediction, parse_json_object


class GroqClient:
    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path = ".cache/groq_claims",
        max_retries: int = 4,
        retry_sleep: float = 2.0,
        use_cache: bool = True,
    ) -> None:
        self.model = model or os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.api_key = os.getenv("GROQ_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = int(os.getenv("GROQ_MAX_RETRIES", str(max_retries)))
        self.retry_sleep = retry_sleep
        self.max_rate_limit_wait = float(os.getenv("GROQ_RATE_LIMIT_MAX_WAIT", "330"))
        self.use_cache = use_cache
        self._client = None
        self._sdk_error = None
        self._init_sdk()

    @property
    def available(self) -> bool:
        return bool(self.api_key and self._client)

    def _init_sdk(self) -> None:
        if not self.api_key:
            self._sdk_error = "GROQ_API_KEY is not set"
            return
        try:
            from groq import Groq  # type: ignore

            self._client = Groq(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover - depends on local environment
            self._sdk_error = f"Groq SDK unavailable: {exc}"

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
                raw_text = self._call_groq(prompt, images)
                parsed = parse_json_object(raw_text)
                self._write_cache(cache_key, parsed, source="groq")
                return parsed
            except Exception as exc:  # pragma: no cover - live API path
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(last_error, attempt))

        result = default_prediction(self._safe_fallback_reason(fallback_reason, last_error), [str(i["id"]) for i in images])
        if not self._is_rate_limit_error(last_error):
            self._write_cache(cache_key, result, source="fallback")
        return result

    def _retry_delay(self, error: str, attempt: int) -> float:
        if self._is_rate_limit_error(error):
            match = re.search(r"try again in (?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", error, flags=re.I)
            if match:
                minutes = float(match.group(1) or 0)
                seconds = float(match.group(2) or 0)
                return min(minutes * 60 + seconds + 5, self.max_rate_limit_wait)
            return min(60 * (attempt + 1), self.max_rate_limit_wait)
        return self.retry_sleep * (attempt + 1)

    def _is_rate_limit_error(self, error: str) -> bool:
        text = error.lower()
        return "429" in text or "rate limit" in text

    def _safe_fallback_reason(self, fallback_reason: str, error: str) -> str:
        text = error.lower()
        if "429" in text or "rate limit" in text:
            return f"{fallback_reason} Provider rate limit was reached."
        if "401" in text or "unauthorized" in text or "api key" in text:
            return f"{fallback_reason} Provider authentication failed."
        if "413" in text or "request too large" in text:
            return f"{fallback_reason} Provider rejected the image request as too large."
        return f"{fallback_reason} Provider response was unavailable or invalid."

    def _call_groq(self, prompt: str, images: List[Dict[str, object]]) -> str:
        assert self._client is not None
        content: List[Dict[str, object]] = [{"type": "text", "text": prompt}]
        for image in images[:5]:
            path = Path(str(image["path"]))
            mime_type = str(image.get("mime_type") or "image/jpeg")
            image_bytes, mime_type = self._read_image_bytes(path, mime_type)
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                }
            )

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_completion_tokens=900,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content or ""

    def _read_image_bytes(self, path: Path, mime_type: str, max_bytes: int = 3_500_000) -> tuple[bytes, str]:
        data = path.read_bytes()
        if len(data) <= max_bytes:
            return data, mime_type
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as img:
                img.thumbnail((1600, 1600))
                rgb = img.convert("RGB")
                buffer = BytesIO()
                rgb.save(buffer, format="JPEG", quality=86, optimize=True)
                compressed = buffer.getvalue()
                if compressed and len(compressed) < len(data):
                    return compressed, "image/jpeg"
                return data, mime_type
        except Exception:
            return data, mime_type

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
