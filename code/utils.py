from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: Iterable[Dict[str, str]], columns: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True) if out_path.parent != Path(".") else None
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def load_history(path: str | Path) -> Dict[str, Dict[str, str]]:
    if not Path(path).exists():
        return {}
    return {row.get("user_id", ""): row for row in read_csv_rows(path)}


def load_requirements(path: str | Path, claim_object: str | None = None) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    rows = read_csv_rows(path)
    if not claim_object:
        return rows
    obj = claim_object.strip().lower()
    return [r for r in rows if r.get("claim_object", "").strip().lower() in {"all", obj}]


def split_image_paths(value: str) -> List[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def resolve_image_path(image_root: str | Path, image_path: str) -> Path:
    raw = Path(image_path)
    if raw.is_absolute():
        return raw
    return Path(image_root) / raw


def image_id_from_path(image_path: str | Path) -> str:
    return Path(str(image_path)).stem


def detect_mime_type(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    suffix = Path(path).suffix.lower()
    if mime:
        return mime
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def collect_images(image_root: str | Path, image_paths_value: str) -> Tuple[List[Dict[str, object]], List[str]]:
    images: List[Dict[str, object]] = []
    missing: List[str] = []
    for rel in split_image_paths(image_paths_value):
        abs_path = resolve_image_path(image_root, rel)
        if not abs_path.exists() or not abs_path.is_file():
            missing.append(rel)
            continue
        images.append(
            {
                "id": image_id_from_path(rel),
                "relative_path": rel,
                "path": abs_path,
                "mime_type": detect_mime_type(abs_path),
            }
        )
    return images, missing


def stable_cache_key(payload: Dict[str, object], image_paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for path in image_paths:
        p = Path(path)
        h.update(str(p).encode("utf-8"))
        try:
            stat = p.stat()
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
        except OSError:
            pass
    return h.hexdigest()
