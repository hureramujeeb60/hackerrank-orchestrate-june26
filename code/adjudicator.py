from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from schema import (
    OBJECT_PARTS,
    csv_bool,
    normalize_prediction,
    normalize_risk_flags,
    normalize_supporting_ids,
    normalize_token,
    parse_bool,
)


ADJUDICATOR_VERSION = "2026-06-20.1"


PART_KEYWORDS = {
    "car": [
        ("rear_bumper", ("rear bumper", "back bumper", "back of the car", "back looks", "rear side")),
        ("front_bumper", ("front bumper", "front side", "front-end", "front end", "bumper ke upar")),
        ("windshield", ("windshield", "front glass", "glass")),
        ("side_mirror", ("side mirror", "mirror")),
        ("headlight", ("headlight", "light")),
        ("taillight", ("taillight", "tail light")),
        ("hood", ("hood", "top panel")),
        ("door", ("door", "door panel")),
        ("fender", ("fender",)),
        ("body", ("body", "side of my car", "side panel")),
    ],
    "laptop": [
        ("hinge", ("hinge",)),
        ("trackpad", ("trackpad",)),
        ("keyboard", ("keyboard", "keys")),
        ("screen", ("screen", "display", "display glass")),
        ("corner", ("corner",)),
        ("lid", ("lid",)),
        ("port", ("port",)),
        ("base", ("base",)),
        ("body", ("body", "outer")),
    ],
    "package": [
        ("contents", ("contents", "item i ordered", "product inside", "not inside", "missing")),
        ("seal", ("seal", "tape", "seal wali", "opened", "open flap", "torn-open")),
        ("package_corner", ("corner",)),
        ("package_side", ("side", "surface", "outside")),
        ("label", ("label",)),
        ("box", ("box", "shipping box", "delivery box", "package")),
        ("item", ("item", "product")),
    ],
}

ISSUE_KEYWORDS = [
    ("missing_part", ("not inside", "missing", "could not find", "contents are missing")),
    ("scratch", ("scratch", "scrape", "scraped", "mark")),
    ("crushed_packaging", ("crushed", "crush")),
    ("water_damage", ("water damaged", "wet", "water damage")),
    ("stain", ("stain", "sticky", "liquid mark")),
    ("torn_packaging", ("torn", "opened", "open", "phati", "seal")),
    ("broken_part", ("broken", "not sitting", "wobbles", "damaged mirror", "component")),
    ("crack", ("crack", "cracked", "shattered", "glass")),
    ("dent", ("dent", "bump", "deformation")),
]

ISSUES_BY_OBJECT = {
    "car": {"dent", "scratch", "crack", "broken_part", "missing_part"},
    "laptop": {"dent", "scratch", "crack", "broken_part", "missing_part", "water_damage", "stain"},
    "package": {"missing_part", "crushed_packaging", "torn_packaging", "water_damage", "stain"},
}


def adjudicate_prediction(
    raw: Dict[str, Any],
    claim: Dict[str, str],
    history: Dict[str, str] | None,
    image_ids: List[str],
) -> Dict[str, str]:
    base = normalize_prediction(raw, claim, image_ids)
    claim_object = normalize_token(claim.get("claim_object", ""))
    claim_text = claim.get("user_claim", "")
    claimed_part, claimed_issue = extract_claim_target(claim_object, claim_text)
    observations = extract_observations(raw, base, image_ids)

    usable = [o for o in observations if o["usable_for_claim"]]
    raw_supported_obs = [o for o in usable if o["claimed_damage_visible"]]
    absent_obs = [o for o in usable if o["claimed_damage_absent"]]
    inferred_mismatches = [
        o
        for o in raw_supported_obs
        if not observation_matches_claim(o, claim_object, claimed_part, claimed_issue)
        or ("text_instruction_present" in o["risk_flags"] and absent_obs)
    ]
    supported_obs = [o for o in raw_supported_obs if o not in inferred_mismatches]
    mismatch_obs = [
        o
        for o in observations
        if any(f in o["risk_flags"] for f in ("claim_mismatch", "wrong_object", "wrong_object_part"))
    ] + inferred_mismatches
    clear_claim_mismatch = [o for o in mismatch_obs if "claim_mismatch" in o["risk_flags"]]
    clear_claim_mismatch.extend([o for o in inferred_mismatches if o not in clear_claim_mismatch])

    risk_flags = merge_risk_flags(base.get("risk_flags"), observation_risks(observations), history_flags(history))
    selected_ids = [o["image_id"] for o in supported_obs or absent_obs or clear_claim_mismatch or usable if o["image_id"]]
    selected_ids_text = normalize_supporting_ids(selected_ids, image_ids)

    final = dict(base)
    if not image_ids:
        final.update(
            {
                "evidence_standard_met": "false",
                "risk_flags": "wrong_object;damage_not_visible;manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            }
        )
        return final

    if claim_object == "package" and (claimed_issue == "missing_part" or claimed_part == "contents"):
        reason = "The package contents are not enough by themselves to verify that a specific item is missing."
        final.update(
            {
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": reason,
                "risk_flags": merge_risk_flags(risk_flags, "cropped_or_obstructed;damage_not_visible;manual_review_required"),
                "issue_type": "unknown",
                "object_part": "contents",
                "claim_status": "not_enough_information",
                "claim_status_justification": reason,
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            }
        )
        return final

    if supported_obs:
        best = supported_obs[0]
        issue = calibrate_issue(best["visible_issue"] or base.get("issue_type"), claim_object, claimed_part, claim_text)
        part = calibrate_part(best["visible_part"] or claimed_part or base.get("object_part"), claim_object)
        final.update(
            {
                "evidence_standard_met": "true",
                "evidence_standard_met_reason": f"The claimed {part} is visible and the claimed damage is visible.",
                "risk_flags": risk_flags,
                "issue_type": issue,
                "object_part": part,
                "claim_status": "supported",
                "claim_status_justification": f"The image evidence supports the claim because {best['image_id']} shows {issue} on the {part}.",
                "supporting_image_ids": selected_ids_text,
                "valid_image": "true",
                "severity": calibrate_severity(best["severity_hint"] or base.get("severity"), issue, "supported", claim_text),
            }
        )
    elif absent_obs and not clear_claim_mismatch:
        best = absent_obs[0]
        part = calibrate_part(best["visible_part"] or claimed_part or base.get("object_part"), claim_object)
        final.update(
            {
                "evidence_standard_met": "true",
                "evidence_standard_met_reason": f"The claimed {part} is visible enough to evaluate and the claimed damage is not visible.",
                "risk_flags": merge_risk_flags(risk_flags, "damage_not_visible"),
                "issue_type": "none",
                "object_part": part,
                "claim_status": "contradicted",
                "claim_status_justification": f"The claimed {part} is visible in {best['image_id']}, but the claimed damage is not visible.",
                "supporting_image_ids": selected_ids_text,
                "valid_image": "true",
                "severity": "none",
            }
        )
    elif clear_claim_mismatch:
        best = clear_claim_mismatch[0]
        multiple_identity_conflict = len(image_ids) > 1 and "wrong_object" in best["risk_flags"]
        wrong_part_support = claimed_part != "unknown" and not part_matches_claim(best["visible_part"], claim_object, claimed_part)
        wrong_issue_with_absent_evidence = (
            claimed_issue != "unknown"
            and not issue_matches_claim(best["visible_issue"], claimed_issue, claim_object, claimed_part)
            and len(image_ids) > 1
            and bool(absent_obs)
        )
        text_instruction_conflict = "text_instruction_present" in best["risk_flags"] and bool(absent_obs)
        if multiple_identity_conflict:
            final.update(
                {
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "The image set has an object or identity mismatch, so the claim cannot be reliably verified.",
                    "risk_flags": merge_risk_flags(risk_flags, "wrong_object;claim_mismatch;manual_review_required"),
                    "issue_type": calibrate_issue(best["visible_issue"] or base.get("issue_type"), claim_object, claimed_part, claim_text),
                    "object_part": calibrate_part(best["visible_part"] or claimed_part or base.get("object_part"), claim_object),
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "The submitted images do not reliably connect the visible damage to the claimed object and part.",
                    "supporting_image_ids": selected_ids_text,
                    "valid_image": "true",
                    "severity": "unknown",
                }
            )
        elif wrong_issue_with_absent_evidence:
            part = calibrate_part(claimed_part or best["visible_part"] or base.get("object_part"), claim_object)
            issue = calibrate_issue(best["visible_issue"] or base.get("issue_type"), claim_object, claimed_part, claim_text)
            final.update(
                {
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "The image set gives conflicting evidence about the claimed damage, so the claim cannot be reliably verified.",
                    "risk_flags": merge_risk_flags(risk_flags, "claim_mismatch;manual_review_required"),
                    "issue_type": issue,
                    "object_part": part,
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "One image appears to show a different issue while another relevant image does not show the claimed damage.",
                    "supporting_image_ids": selected_ids_text,
                    "valid_image": "true",
                    "severity": "unknown",
                }
            )
        elif wrong_part_support or text_instruction_conflict:
            part = calibrate_part(claimed_part or base.get("object_part"), claim_object)
            final.update(
                {
                    "evidence_standard_met": "true",
                    "evidence_standard_met_reason": f"The claimed {part} is visible enough to evaluate, but the claimed damage is not reliable or not visible.",
                    "risk_flags": merge_risk_flags(risk_flags, "damage_not_visible;claim_mismatch;manual_review_required"),
                    "issue_type": "none",
                    "object_part": part,
                    "claim_status": "contradicted",
                    "claim_status_justification": f"The evidence does not show the claimed damage on the {part}.",
                    "supporting_image_ids": selected_ids_text,
                    "valid_image": "true",
                    "severity": "none",
                }
            )
        else:
            final.update(
                {
                    "evidence_standard_met": "true",
                    "evidence_standard_met_reason": "The visible evidence is sufficient to evaluate and does not match the claim.",
                    "risk_flags": merge_risk_flags(risk_flags, "claim_mismatch;manual_review_required"),
                    "issue_type": calibrate_issue(best["visible_issue"] or base.get("issue_type"), claim_object, claimed_part, claim_text),
                    "object_part": calibrate_part(best["visible_part"] or base.get("object_part"), claim_object),
                    "claim_status": "contradicted",
                    "claim_status_justification": "The image evidence shows a different condition than the one described in the claim.",
                    "supporting_image_ids": selected_ids_text,
                    "valid_image": "true",
                    "severity": calibrate_severity(best["severity_hint"] or base.get("severity"), best["visible_issue"] or base.get("issue_type"), "contradicted", claim_text),
                }
            )
    elif base["claim_status"] == "contradicted" and base["evidence_standard_met"] == "true":
        final["severity"] = calibrate_severity(base.get("severity"), base.get("issue_type"), "contradicted", claim_text)
    else:
        part = claimed_part if claimed_part and claimed_part in OBJECT_PARTS.get(claim_object, set()) else calibrate_part(base.get("object_part"), claim_object)
        final.update(
            {
                "evidence_standard_met": "false",
                "risk_flags": merge_risk_flags(risk_flags, "damage_not_visible;manual_review_required" if risk_flags == "none" else risk_flags),
                "issue_type": "unknown",
                "object_part": part if part != "unknown" else "unknown",
                "claim_status": "not_enough_information",
                "supporting_image_ids": "none",
                "severity": "unknown",
            }
        )

    final["risk_flags"] = normalize_risk_flags(final.get("risk_flags"))
    final["supporting_image_ids"] = normalize_supporting_ids(final.get("supporting_image_ids"), image_ids)
    if final["claim_status"] == "not_enough_information":
        final["severity"] = "unknown"
    if final["issue_type"] == "none":
        final["severity"] = "none"
    return normalize_prediction(final, claim, image_ids)


def extract_claim_target(claim_object: str, claim_text: str) -> Tuple[str, str]:
    focus = _plain(_last_customer_text(claim_text))
    text = focus or _plain(claim_text)
    full_text = _plain(claim_text)
    part = "unknown"
    for candidate, phrases in PART_KEYWORDS.get(claim_object, []):
        if any(phrase in text for phrase in phrases):
            part = candidate
            break
    if part == "unknown":
        for candidate, phrases in PART_KEYWORDS.get(claim_object, []):
            if any(phrase in full_text for phrase in phrases):
                part = candidate
                break

    issue = "unknown"
    allowed_issues = ISSUES_BY_OBJECT.get(claim_object, {candidate for candidate, _ in ISSUE_KEYWORDS})
    for candidate, phrases in ISSUE_KEYWORDS:
        if candidate not in allowed_issues:
            continue
        if any(phrase in text for phrase in phrases):
            issue = candidate
            break
    if issue == "unknown":
        for candidate, phrases in ISSUE_KEYWORDS:
            if candidate not in allowed_issues:
                continue
            if any(phrase in full_text for phrase in phrases):
                issue = candidate
                break

    if claim_object == "package" and "missing" in text and any(phrase in text for phrase in ("not", "nahi", "sirf", "only")):
        if any(phrase in text for phrase in ("torn", "opened", "open", "seal", "phati")):
            issue = "torn_packaging"
            part = "seal" if "seal" in text or "seal" in full_text or "torn packaging" in text else part
    if claim_object == "laptop" and "screen" in text and "not" in text and any(phrase in text for phrase in ("hinge", "keyboard")):
        part = "screen"

    if claim_object == "laptop" and part == "keyboard" and issue == "water_damage":
        issue = "stain"
    if claim_object in {"car", "laptop"} and part in {"screen", "windshield"} and issue == "glass_shatter":
        issue = "crack"
    return part, issue


def extract_observations(raw: Dict[str, Any], base: Dict[str, str], image_ids: List[str]) -> List[Dict[str, Any]]:
    candidates = raw.get("observations") or raw.get("image_observations") or raw.get("per_image_observations") or []
    if isinstance(candidates, dict):
        candidates = candidates.get("images", [])
    observations = [_normalize_observation(item, image_ids) for item in candidates if isinstance(item, dict)]
    observations = [item for item in observations if item["image_id"] in image_ids]
    if observations:
        return observations

    status = base.get("claim_status")
    issue = base.get("issue_type")
    part = base.get("object_part")
    ids = normalize_supporting_ids(base.get("supporting_image_ids"), image_ids)
    selected = [x for x in ids.split(";") if x and x != "none"] or image_ids[:1]
    return [
        {
            "image_id": selected[0],
            "visible_part": part,
            "visible_issue": issue,
            "claimed_damage_visible": status == "supported" and issue not in {"none", "unknown"},
            "claimed_damage_absent": status == "contradicted" and issue == "none",
            "usable_for_claim": base.get("evidence_standard_met") == "true",
            "risk_flags": set(normalize_risk_flags(base.get("risk_flags")).split(";")) - {"none"},
            "severity_hint": base.get("severity"),
        }
    ]


def _normalize_observation(item: Dict[str, Any], image_ids: List[str]) -> Dict[str, Any]:
    image_id = str(item.get("image_id") or item.get("id") or "").strip()
    if image_id not in image_ids:
        image_id = normalize_supporting_ids(image_id, image_ids).split(";")[0]
        if image_id == "none":
            image_id = ""
    risk_flags = merge_risk_flags(item.get("quality_flags"), item.get("risk_flags"), item.get("mismatch_flags"))
    visible_part = normalize_token(item.get("visible_part") or item.get("object_part") or item.get("part"))
    visible_issue = normalize_token(item.get("visible_issue") or item.get("issue_type") or item.get("damage_type"))
    damage_present = parse_bool(item.get("claimed_damage_visible"), False) or parse_bool(item.get("damage_present"), False)
    damage_absent = parse_bool(item.get("claimed_damage_absent"), False) or parse_bool(item.get("damage_absent"), False)
    usable = parse_bool(item.get("usable_for_claim"), False) or parse_bool(item.get("claimed_part_visible"), False)
    if damage_present:
        usable = True
    return {
        "image_id": image_id,
        "visible_part": visible_part,
        "visible_issue": visible_issue,
        "claimed_damage_visible": damage_present,
        "claimed_damage_absent": damage_absent,
        "usable_for_claim": usable,
        "risk_flags": set(normalize_risk_flags(risk_flags).split(";")) - {"none"},
        "severity_hint": normalize_token(item.get("severity_hint") or item.get("severity")),
    }


def observation_risks(observations: Iterable[Dict[str, Any]]) -> str:
    flags: List[str] = []
    for obs in observations:
        flags.extend(sorted(obs.get("risk_flags", set())))
    return merge_risk_flags(flags)


def history_flags(history: Dict[str, str] | None) -> str:
    flags = normalize_risk_flags((history or {}).get("history_flags"))
    if "user_history_risk" in flags.split(";") or "manual_review_required" in flags.split(";"):
        flags = merge_risk_flags(flags, "manual_review_required")
    return flags


def observation_matches_claim(obs: Dict[str, Any], claim_object: str, claimed_part: str, claimed_issue: str) -> bool:
    return part_matches_claim(obs.get("visible_part"), claim_object, claimed_part) and issue_matches_claim(
        obs.get("visible_issue"), claimed_issue, claim_object, claimed_part
    )


def part_matches_claim(visible_part: Any, claim_object: str, claimed_part: str) -> bool:
    if claimed_part in {"", "unknown"}:
        return True
    visible = normalize_token(visible_part)
    if visible == claimed_part:
        return True
    if claim_object == "package" and {visible, claimed_part} <= {"seal", "package_side", "box"}:
        return True
    if claim_object == "car" and claimed_part == "door" and visible == "body":
        return True
    return False


def issue_matches_claim(visible_issue: Any, claimed_issue: str, claim_object: str, claimed_part: str) -> bool:
    if claimed_issue in {"", "unknown"}:
        return True
    visible = normalize_token(visible_issue)
    if visible == claimed_issue:
        return True
    if {visible, claimed_issue} <= {"crack", "glass_shatter"} and claimed_part in {"screen", "windshield"}:
        return True
    if claim_object == "package" and {visible, claimed_issue} <= {"torn_packaging", "broken_part"}:
        return True
    if claim_object == "package" and claimed_issue == "water_damage" and visible in {"water_damage", "stain"}:
        return True
    if claim_object == "laptop" and claimed_part == "keyboard" and {visible, claimed_issue} <= {"water_damage", "stain"}:
        return True
    if claimed_issue == "broken_part" and visible in {"broken_part", "crack", "dent", "scratch"}:
        return True
    return False


def merge_risk_flags(*values: Any) -> str:
    flags: List[str] = []
    for value in values:
        normalized = normalize_risk_flags(value)
        for flag in normalized.split(";"):
            if flag and flag != "none" and flag not in flags:
                flags.append(flag)
    return ";".join(flags) if flags else "none"


def calibrate_part(value: Any, claim_object: str) -> str:
    token = normalize_token(value)
    allowed = OBJECT_PARTS.get(claim_object, {"unknown"})
    return token if token in allowed else "unknown"


def calibrate_issue(value: Any, claim_object: str, claimed_part: str, claim_text: str) -> str:
    issue = normalize_token(value)
    text = _plain(claim_text)
    if issue == "glass_shatter" and claim_object in {"car", "laptop"} and claimed_part in {"screen", "windshield"}:
        return "glass_shatter" if "shatter" in text and "crack" not in text else "crack"
    if issue == "water_damage" and claim_object == "laptop" and claimed_part == "keyboard":
        return "stain"
    if issue == "stain" and claim_object == "package" and "water" in _plain(claim_text):
        return "water_damage"
    if claimed_part in {"side_mirror", "hinge"} and issue in {"crack", "dent", "scratch"} and "broken" in text:
        return "broken_part"
    if issue in {"dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part", "torn_packaging", "crushed_packaging", "water_damage", "stain", "none"}:
        return issue
    return "unknown"


def calibrate_severity(value: Any, issue: Any, status: str, claim_text: str) -> str:
    issue_token = normalize_token(issue)
    severity = normalize_token(value)
    if status == "not_enough_information":
        return "unknown"
    if issue_token == "none":
        return "none"
    if issue_token in {"scratch", "dent", "stain", "crushed_packaging", "torn_packaging"}:
        return "medium" if any(word in _plain(claim_text) for word in ("badly", "pretty bad", "shattered")) else "low"
    if issue_token in {"crack", "broken_part", "water_damage"}:
        return "medium" if severity in {"unknown", "none", "low", ""} else severity
    if issue_token == "missing_part":
        return "high"
    return severity if severity in {"low", "medium", "high", "none", "unknown"} else "unknown"


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _last_customer_text(text: str) -> str:
    matches = re.findall(r"customer:\s*(.*?)(?=\|\s*support:|$)", text, flags=re.I | re.S)
    return matches[-1].strip() if matches else text
