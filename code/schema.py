from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List


REQUIRED_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

PREDICTION_COLUMNS = REQUIRED_COLUMNS[4:]

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
SEVERITIES = {"none", "low", "medium", "high", "unknown"}
RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
OBJECT_PARTS = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

VALUE_ALIASES = {
    "yes": "true",
    "no": "false",
    "not enough information": "not_enough_information",
    "insufficient": "not_enough_information",
    "insufficient_evidence": "not_enough_information",
    "not_enough_evidence": "not_enough_information",
    "supports": "supported",
    "support": "supported",
    "contradicts": "contradicted",
    "contradict": "contradicted",
    "broken": "broken_part",
    "breakage": "broken_part",
    "damaged": "broken_part",
    "physical_damage": "broken_part",
    "missing": "missing_part",
    "missing_item": "missing_part",
    "missing_contents": "missing_part",
    "shattered_glass": "glass_shatter",
    "shattered": "glass_shatter",
    "water": "water_damage",
    "wet": "water_damage",
    "wet_stain": "water_damage",
    "torn": "torn_packaging",
    "opened": "torn_packaging",
    "open": "torn_packaging",
    "crushed": "crushed_packaging",
    "bumper_front": "front_bumper",
    "bumper_rear": "rear_bumper",
    "mirror": "side_mirror",
    "light": "headlight",
    "screen_crack": "crack",
}


def normalize_token(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"[\s/-]+", "_", text)
    text = re.sub(r"[^a-z0-9_;,]+", "", text)
    return VALUE_ALIASES.get(text, text)


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_token(value)
    if text in {"true", "1", "y", "yes"}:
        return True
    if text in {"false", "0", "n", "no"}:
        return False
    return default


def csv_bool(value: bool) -> str:
    return "true" if value else "false"


def clamp_enum(value: Any, allowed: set[str], default: str) -> str:
    token = normalize_token(value)
    return token if token in allowed else default


def normalize_risk_flags(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        raw_parts = value
    else:
        raw_parts = re.split(r"[;,]", str(value))
    flags: List[str] = []
    for part in raw_parts:
        flag = clamp_enum(part, RISK_FLAGS, "")
        if flag and flag != "none" and flag not in flags:
            flags.append(flag)
    return ";".join(flags) if flags else "none"


def normalize_supporting_ids(value: Any, valid_ids: Iterable[str]) -> str:
    valid = set(valid_ids)
    if value is None:
        return "none"
    if isinstance(value, list):
        raw_parts = value
    else:
        raw_parts = re.split(r"[;,]", str(value))
    ids: List[str] = []
    for part in raw_parts:
        token = str(part).strip()
        token = re.sub(r"\.(jpg|jpeg|png|webp)$", "", token, flags=re.I)
        token = token.split("/")[-1].split("\\")[-1]
        if token in valid and token not in ids:
            ids.append(token)
    return ";".join(ids) if ids else "none"


def default_prediction(reason: str, image_ids: Iterable[str] | None = None) -> Dict[str, str]:
    ids = list(image_ids or [])
    valid = bool(ids)
    risk = "manual_review_required"
    if not valid:
        risk = "wrong_object;damage_not_visible;manual_review_required"
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason[:300] or "The image evidence is not sufficient for automated review.",
        "risk_flags": risk,
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": reason[:300] or "The submitted evidence is insufficient to verify the claim.",
        "supporting_image_ids": "none",
        "valid_image": csv_bool(valid),
        "severity": "unknown",
    }


def normalize_prediction(raw: Dict[str, Any], claim: Dict[str, str], image_ids: List[str]) -> Dict[str, str]:
    prediction = default_prediction("Model output was incomplete or invalid.", image_ids)
    prediction.update({k: raw.get(k, prediction[k]) for k in PREDICTION_COLUMNS if isinstance(raw, dict)})

    claim_object = normalize_token(claim.get("claim_object", ""))
    part_allowed = OBJECT_PARTS.get(claim_object, {"unknown"})

    evidence_met = parse_bool(prediction.get("evidence_standard_met"), False)
    valid_image = parse_bool(prediction.get("valid_image"), bool(image_ids))
    status = clamp_enum(prediction.get("claim_status"), CLAIM_STATUSES, "not_enough_information")
    issue_type = clamp_enum(prediction.get("issue_type"), ISSUE_TYPES, "unknown")
    object_part = clamp_enum(prediction.get("object_part"), part_allowed, "unknown")
    severity = clamp_enum(prediction.get("severity"), SEVERITIES, "unknown")
    risk_flags = normalize_risk_flags(prediction.get("risk_flags"))
    supporting_ids = normalize_supporting_ids(prediction.get("supporting_image_ids"), image_ids)

    if not valid_image:
        evidence_met = False
        status = "not_enough_information"
        supporting_ids = "none"
        if risk_flags == "none":
            risk_flags = "damage_not_visible;manual_review_required"

    if not evidence_met and status == "supported":
        status = "not_enough_information"
        supporting_ids = "none"
        if risk_flags == "none":
            risk_flags = "manual_review_required"

    if status == "supported":
        evidence_met = True
        valid_image = True
        if supporting_ids == "none" and image_ids:
            supporting_ids = image_ids[0]

    if issue_type == "none":
        severity = "none"

    reason = str(prediction.get("evidence_standard_met_reason") or "").strip()
    justification = str(prediction.get("claim_status_justification") or "").strip()
    if not reason:
        reason = "The image evidence was reviewed against the claim and evidence rules."
    if not justification:
        justification = "The final status is based on the visible image evidence and claim details."

    return {
        "evidence_standard_met": csv_bool(evidence_met),
        "evidence_standard_met_reason": reason[:500],
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": status,
        "claim_status_justification": justification[:500],
        "supporting_image_ids": supporting_ids,
        "valid_image": csv_bool(valid_image),
        "severity": severity,
    }


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    return data


def required_header() -> List[str]:
    return list(REQUIRED_COLUMNS)
