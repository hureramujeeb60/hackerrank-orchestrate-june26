from __future__ import annotations

import re
from pathlib import Path
from typing import Any


INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]

OUTPUT_COLUMNS = [
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

SEVERITIES = {"none", "low", "medium", "high", "unknown"}

RISK_ORDER = [
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
]

ISSUES_BY_OBJECT = {
    "car": {
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "stain",
        "none",
        "unknown",
    },
    "laptop": {
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "stain",
        "water_damage",
        "none",
        "unknown",
    },
    "package": {
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
        "stain",
        "none",
        "unknown",
    },
}

ALIASES = {
    "claim_status": {
        "support": "supported",
        "supports": "supported",
        "supported_by_images": "supported",
        "verified": "supported",
        "contradict": "contradicted",
        "contradicts": "contradicted",
        "not_supported": "contradicted",
        "insufficient": "not_enough_information",
        "insufficient_evidence": "not_enough_information",
        "unknown": "not_enough_information",
        "not_enough_info": "not_enough_information",
        "cannot_determine": "not_enough_information",
    },
    "issue_type": {
        "shatter": "glass_shatter",
        "shattered_glass": "glass_shatter",
        "broken": "broken_part",
        "broken component": "broken_part",
        "missing": "missing_part",
        "tear": "torn_packaging",
        "torn": "torn_packaging",
        "crushed": "crushed_packaging",
        "water": "water_damage",
        "no_damage": "none",
        "no_visible_damage": "none",
    },
    "severity": {
        "n/a": "none",
        "na": "none",
        "no_damage": "none",
        "minor": "low",
        "moderate": "medium",
        "severe": "high",
    },
}


def canonical_token(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = text.replace("-", "_").replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text


def normalize_bool(value: Any, default: str = "false") -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    token = canonical_token(value)
    if token in {"true", "yes", "y", "1", "usable", "valid"}:
        return "true"
    if token in {"false", "no", "n", "0", "unusable", "invalid"}:
        return "false"
    return default


def normalize_enum(value: Any, allowed: set[str], default: str, kind: str | None = None) -> str:
    token = canonical_token(value)
    if kind and token in ALIASES.get(kind, {}):
        token = ALIASES[kind][token]
    return token if token in allowed else default


def normalize_object_part(value: Any, claim_object: str) -> str:
    token = canonical_token(value)
    token = {
        "bumper_front": "front_bumper",
        "front": "front_bumper",
        "bumper_rear": "rear_bumper",
        "rear": "rear_bumper",
        "mirror": "side_mirror",
        "light": "headlight",
        "panel": "body",
        "chassis": "body",
        "display": "screen",
        "touchpad": "trackpad",
        "cover": "lid",
        "corner_of_box": "package_corner",
        "side": "package_side",
        "flap": "seal",
        "inside": "contents",
        "content": "contents",
    }.get(token, token)
    return token if token in OBJECT_PARTS.get(claim_object, {"unknown"}) else "unknown"


def split_semicolon(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        pieces = value
    else:
        pieces = re.split(r"[;,]", str(value))
    return [str(piece).strip() for piece in pieces if str(piece).strip()]


def normalize_risk_flags(value: Any, history_flags: str | None = None) -> str:
    flags = []
    for source in (value, history_flags):
        for piece in split_semicolon(source):
            token = canonical_token(piece)
            if token in RISK_FLAGS and token != "none":
                flags.append(token)
    unique = {flag for flag in flags if flag in RISK_FLAGS}
    if not unique:
        return "none"
    return ";".join(flag for flag in RISK_ORDER if flag in unique)


def normalize_supporting_image_ids(value: Any, allowed_ids: list[str]) -> str:
    allowed = set(allowed_ids)
    ids = []
    for piece in split_semicolon(value):
        token = Path(piece).stem.strip()
        if token in {"", "none", "unknown", "null", "n_a", "na"}:
            continue
        if allowed and token not in allowed:
            continue
        ids.append(token)
    unique = []
    for image_id in ids:
        if image_id not in unique:
            unique.append(image_id)
    return ";".join(unique) if unique else "none"


def infer_issue_type(text: str) -> str:
    lowered = text.lower()
    for needle, label in [
        ("shatter", "glass_shatter"),
        ("crack", "crack"),
        ("scratch", "scratch"),
        ("scrape", "scratch"),
        ("dent", "dent"),
        ("broken", "broken_part"),
        ("missing", "missing_part"),
        ("torn", "torn_packaging"),
        ("tear", "torn_packaging"),
        ("crush", "crushed_packaging"),
        ("water", "water_damage"),
        ("wet", "water_damage"),
        ("stain", "stain"),
    ]:
        if needle in lowered:
            return label
    return "unknown"


def infer_object_part(text: str, claim_object: str) -> str:
    lowered = text.lower()
    object_terms = {
        "car": [
            ("front bumper", "front_bumper"),
            ("rear bumper", "rear_bumper"),
            ("back bumper", "rear_bumper"),
            ("bumper", "body"),
            ("door", "door"),
            ("hood", "hood"),
            ("windshield", "windshield"),
            ("front glass", "windshield"),
            ("mirror", "side_mirror"),
            ("headlight", "headlight"),
            ("taillight", "taillight"),
            ("tail light", "taillight"),
            ("fender", "fender"),
            ("quarter", "quarter_panel"),
            ("body", "body"),
        ],
        "laptop": [
            ("screen", "screen"),
            ("keyboard", "keyboard"),
            ("trackpad", "trackpad"),
            ("touchpad", "trackpad"),
            ("hinge", "hinge"),
            ("lid", "lid"),
            ("corner", "corner"),
            ("port", "port"),
            ("base", "base"),
            ("body", "body"),
        ],
        "package": [
            ("corner", "package_corner"),
            ("side", "package_side"),
            ("seal", "seal"),
            ("label", "label"),
            ("contents", "contents"),
            ("inside", "contents"),
            ("item", "item"),
            ("box", "box"),
            ("package", "box"),
        ],
    }
    for needle, label in object_terms.get(claim_object, []):
        if needle in lowered:
            return label
    return "unknown"


def calibrate_issue_type(issue_type: str, claim_object: str, object_part: str, text: str) -> str:
    lowered = text.lower()
    if issue_type == "glass_shatter":
        if claim_object == "laptop" and object_part == "screen":
            return "crack"
        if claim_object == "car" and object_part == "side_mirror":
            return "broken_part"
        if "shatter" not in lowered and "shattered" not in lowered:
            return "crack"
    if issue_type not in ISSUES_BY_OBJECT.get(claim_object, ISSUE_TYPES):
        inferred = infer_issue_type(text)
        return inferred if inferred in ISSUES_BY_OBJECT.get(claim_object, ISSUE_TYPES) else "unknown"
    return issue_type


def calibrate_object_part(object_part: str, claim_object: str, text: str) -> str:
    inferred = infer_object_part(text, claim_object)
    generic_parts = {
        "car": {"body", "unknown"},
        "laptop": {"body", "unknown"},
        "package": {"box", "package_side", "unknown"},
    }
    if inferred != "unknown" and object_part in generic_parts.get(claim_object, {"unknown"}):
        return inferred
    return object_part


def short_text(value: Any, fallback: str, max_len: int = 240) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        text = fallback
    return text[:max_len].rstrip()


def normalize_prediction(
    claim: dict[str, str],
    prediction: dict[str, Any] | None,
    image_ids: list[str],
    history: dict[str, str] | None = None,
) -> dict[str, str]:
    prediction = prediction or {}
    history = history or {}
    claim_object = canonical_token(claim.get("claim_object")) or "unknown"
    if claim_object not in OBJECT_PARTS:
        claim_object = "unknown"

    row = {column: claim.get(column, "") for column in INPUT_COLUMNS}
    row["claim_object"] = claim_object if claim_object != "unknown" else claim.get("claim_object", "")

    status = normalize_enum(
        prediction.get("claim_status"),
        CLAIM_STATUSES,
        "not_enough_information",
        "claim_status",
    )
    issue_type = normalize_enum(prediction.get("issue_type"), ISSUE_TYPES, "unknown", "issue_type")
    object_part = normalize_object_part(prediction.get("object_part"), claim_object)
    severity = normalize_enum(prediction.get("severity"), SEVERITIES, "unknown", "severity")
    evidence_standard_met = normalize_bool(prediction.get("evidence_standard_met"), "false")
    valid_image = normalize_bool(prediction.get("valid_image"), "false")
    risk_flags = normalize_risk_flags(prediction.get("risk_flags"), history.get("history_flags"))
    supporting_image_ids = normalize_supporting_image_ids(
        prediction.get("supporting_image_ids"),
        image_ids,
    )

    evidence_reason_text = short_text(
        prediction.get("evidence_standard_met_reason"),
        "The image evidence could not be confidently verified by the automated reviewer.",
    )
    justification_text = short_text(
        prediction.get("claim_status_justification"),
        "The submitted evidence is insufficient for a confident automated decision.",
    )
    inference_text = " ".join(
        [
            claim.get("user_claim", ""),
            evidence_reason_text,
            justification_text,
        ]
    )
    if issue_type == "unknown":
        issue_type = infer_issue_type(inference_text)
    if object_part == "unknown":
        object_part = infer_object_part(inference_text, claim_object)
    object_part = calibrate_object_part(object_part, claim_object, inference_text)
    issue_type = calibrate_issue_type(issue_type, claim_object, object_part, inference_text)

    if status == "supported" and issue_type == "missing_part" and object_part == "contents":
        status = "not_enough_information"
        evidence_standard_met = "false"
        supporting_image_ids = "none"
        severity = "unknown"
        if risk_flags == "none":
            risk_flags = "manual_review_required"

    if status == "supported":
        evidence_standard_met = "true"
        valid_image = "true" if valid_image == "false" and image_ids else valid_image
        if supporting_image_ids == "none" and image_ids:
            supporting_image_ids = image_ids[0]
    elif evidence_standard_met == "false" and status == "supported":
        status = "not_enough_information"

    if status == "not_enough_information" and severity not in {"none", "unknown"}:
        severity = "unknown"
    if status == "not_enough_information" and issue_type == "none":
        issue_type = "unknown"
        severity = "unknown"
    if issue_type == "none":
        severity = "none"

    if valid_image == "false" and risk_flags == "none":
        risk_flags = "manual_review_required"

    row.update(
        {
            "evidence_standard_met": evidence_standard_met,
            "evidence_standard_met_reason": evidence_reason_text,
            "risk_flags": risk_flags,
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": status,
            "claim_status_justification": justification_text,
            "supporting_image_ids": supporting_image_ids,
            "valid_image": valid_image,
            "severity": severity,
        }
    )
    return {column: row.get(column, "") for column in OUTPUT_COLUMNS}


def fallback_prediction(
    claim: dict[str, str],
    image_ids: list[str],
    missing_images: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    text = f"{claim.get('user_claim', '')} {claim.get('claim_object', '')}".lower()
    issue = infer_issue_type(text)

    claim_object = canonical_token(claim.get("claim_object"))
    part = infer_object_part(text, claim_object)

    flags = "manual_review_required"
    valid_image = "true" if image_ids and not missing_images else "false"
    if missing_images:
        flags = "manual_review_required"
    if not image_ids:
        flags = "damage_not_visible;manual_review_required"

    why = reason or "The vision model was unavailable or returned an unusable response; using conservative fallback."
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": why,
        "risk_flags": flags,
        "issue_type": issue,
        "object_part": part,
        "claim_status": "not_enough_information",
        "claim_status_justification": "Automated visual verification was not completed, so the claim cannot be supported or contradicted.",
        "supporting_image_ids": "none",
        "valid_image": valid_image,
        "severity": "unknown",
    }
