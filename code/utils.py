from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass
class ImagePayload:
    image_id: str
    display_path: str
    path: Path
    mime_type: str
    data: bytes


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path("") else None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def split_image_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [piece.strip() for piece in str(value).split(";") if piece.strip()]


def image_id_from_path(path_text: str) -> str:
    return Path(path_text.replace("\\", "/")).stem


def resolve_image_path(path_text: str, image_root: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(image_root) / path


def detect_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in {"image/jpeg", "image/png", "image/webp"}:
        return guessed
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def maybe_compress_image(path: Path, max_bytes: int = 2_800_000, max_dimension: int = 2048) -> tuple[bytes, str]:
    data = path.read_bytes()
    try:
        from PIL import Image, ImageOps
    except Exception:
        return data, detect_mime_type(path)

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((max_dimension, max_dimension))
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, (255, 255, 255))
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            for quality in (90, 85, 80, 75, 70):
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                converted = buffer.getvalue()
                if len(converted) <= max_bytes or quality == 70:
                    return converted, "image/jpeg"
    except Exception:
        return data, detect_mime_type(path)


def load_images(image_paths: list[str], image_root: str | Path) -> tuple[list[ImagePayload], list[str]]:
    images: list[ImagePayload] = []
    missing: list[str] = []
    for path_text in image_paths:
        path = resolve_image_path(path_text, image_root)
        if not path.exists() or not path.is_file():
            missing.append(path_text)
            continue
        data, mime_type = maybe_compress_image(path)
        if not mime_type.startswith("image/"):
            missing.append(path_text)
            continue
        images.append(
            ImagePayload(
                image_id=image_id_from_path(path_text),
                display_path=path_text,
                path=path,
                mime_type=mime_type,
                data=data,
            )
        )
    return images, missing


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key, "")}


def select_requirements(claim: dict[str, str], requirements: list[dict[str, str]]) -> list[dict[str, str]]:
    claim_object = claim.get("claim_object", "").strip().lower()
    text = f"{claim.get('user_claim', '')} {claim_object}".lower()
    selected = []
    for requirement in requirements:
        req_object = requirement.get("claim_object", "").strip().lower()
        applies_to = requirement.get("applies_to", "").lower()
        if req_object not in {"all", claim_object}:
            continue
        if req_object == "all":
            selected.append(requirement)
            continue
        keywords = [piece.strip() for piece in applies_to.replace(",", " or ").split("or") if piece.strip()]
        if not keywords or any(keyword in text for keyword in keywords):
            selected.append(requirement)
    return selected[:6]


def stable_claim_key(
    claim: dict[str, str],
    history: dict[str, str] | None,
    requirements: list[dict[str, str]],
    images: list[ImagePayload],
    model: str,
    prompt_version: str,
) -> str:
    payload: dict[str, Any] = {
        "claim": claim,
        "history": history or {},
        "requirements": requirements,
        "images": [
            {
                "display_path": image.display_path,
                "image_id": image.image_id,
                "size": len(image.data),
                "sha256": hashlib.sha256(image.data).hexdigest(),
            }
            for image in images
        ],
        "model": model,
        "prompt_version": prompt_version,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
