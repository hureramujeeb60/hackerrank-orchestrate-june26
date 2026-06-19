from __future__ import annotations

import json

from schema import ISSUE_TYPES, OBJECT_PARTS, RISK_FLAGS, SEVERITIES
from utils import ImagePayload


PROMPT_VERSION = "strict_reviewer_v1"


def build_reviewer_prompt(
    claim: dict[str, str],
    history: dict[str, str] | None,
    requirements: list[dict[str, str]],
    images: list[ImagePayload],
) -> str:
    image_context = [
        {"image_id": image.image_id, "path": image.display_path, "mime_type": image.mime_type}
        for image in images
    ]
    object_part_allowed = OBJECT_PARTS.get(claim.get("claim_object", "").strip().lower(), {"unknown"})
    payload = {
        "claim": {
            "user_id": claim.get("user_id", ""),
            "claim_object": claim.get("claim_object", ""),
            "user_claim": claim.get("user_claim", ""),
            "image_ids": image_context,
        },
        "user_history": history or {},
        "evidence_requirements": requirements,
    }
    return f"""You are a visual evidence reviewer for damage claims.
Return JSON only. Do not include markdown, commentary, or chain-of-thought.

Images are the primary source of truth. The user claim defines what to check.
User history may affect risk_flags and justification, but must not decide claim_status by itself.

Decision rules:
- If the claimed object or claimed part is not visible, claim_status must be "not_enough_information".
- If the claimed part is visible and claimed damage is visible, claim_status must be "supported".
- If the claimed part is visible and claimed damage is clearly absent, claim_status must be "contradicted".
- If image quality, angle, object identity, or damage visibility prevents verification, use "not_enough_information".
- supporting_image_ids must use image IDs like "img_1", never full paths. Use "none" when no image supports the decision.

Allowed values:
- claim_status: supported, contradicted, not_enough_information
- issue_type: {", ".join(sorted(ISSUE_TYPES))}
- object_part for this claim_object: {", ".join(sorted(object_part_allowed))}
- risk_flags: {", ".join(sorted(RISK_FLAGS))}
- severity: {", ".join(sorted(SEVERITIES))}

Required JSON object:
{{
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "short evidence reason",
  "risk_flags": "semicolon-separated flags or none",
  "issue_type": "allowed issue_type",
  "object_part": "allowed object_part",
  "claim_status": "supported|contradicted|not_enough_information",
  "claim_status_justification": "short image-grounded explanation",
  "supporting_image_ids": "semicolon-separated image IDs or none",
  "valid_image": true or false,
  "severity": "none|low|medium|high|unknown"
}}

Review payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
