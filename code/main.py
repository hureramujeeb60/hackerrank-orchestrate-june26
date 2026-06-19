from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from adjudicator import ADJUDICATOR_VERSION, adjudicate_prediction
from prompts import PROMPT_VERSION, build_prompt
from schema import REQUIRED_COLUMNS, default_prediction, normalize_prediction
from utils import (
    collect_images,
    load_dotenv,
    load_history,
    load_requirements,
    read_csv_rows,
    stable_cache_key,
    write_csv_rows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multimodal claim verification with a vision model.")
    parser.add_argument("--claims", default="dataset/claims.csv", help="Input claims CSV")
    parser.add_argument("--history", default="dataset/user_history.csv", help="User history CSV")
    parser.add_argument("--requirements", default="dataset/evidence_requirements.csv", help="Evidence requirements CSV")
    parser.add_argument("--image-root", default="dataset", help="Root used to resolve image paths")
    parser.add_argument("--output", default="output.csv", help="Output CSV path")
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq", help="Vision model provider")
    parser.add_argument("--cache-dir", default="", help="Local response cache directory")
    parser.add_argument("--strategy", choices=["basic", "structured"], default="structured", help="Prompt strategy")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests")
    parser.add_argument("--no-cache", action="store_true", help="Disable response cache")
    return parser


def predict_claim(
    claim: Dict[str, str],
    history_by_user: Dict[str, Dict[str, str]],
    requirements_path: str,
    image_root: str,
    client: object,
    strategy: str,
) -> Dict[str, str]:
    images, missing_images = collect_images(image_root, claim.get("image_paths", ""))
    image_ids = [str(image["id"]) for image in images]

    if not images:
        raw = default_prediction("No usable submitted images were found for this claim.", image_ids)
        return {**claim, **normalize_prediction(raw, claim, image_ids)}

    history = history_by_user.get(claim.get("user_id", ""), {})
    requirements = load_requirements(requirements_path, claim.get("claim_object"))
    prompt = build_prompt(claim, history, requirements, image_ids, strategy=strategy)
    cache_payload = {
        "claim": claim,
        "history": history,
        "requirements": requirements,
        "image_ids": image_ids,
        "missing_images": missing_images,
        "strategy": strategy,
        "provider": client.__class__.__name__,
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "adjudicator_version": ADJUDICATOR_VERSION,
        "prompt": prompt,
    }
    cache_key = stable_cache_key(cache_payload, [Path(str(i["path"])) for i in images])
    fallback_reason = "Automated visual review could not be completed."
    if missing_images:
        fallback_reason += f" Missing image paths: {', '.join(missing_images)}."
    raw = client.generate_json(prompt, images, cache_key, fallback_reason)
    prediction = adjudicate_prediction(raw, claim, history, image_ids)

    if missing_images and prediction["risk_flags"] == "none":
        prediction["risk_flags"] = "cropped_or_obstructed;manual_review_required"
    return {**claim, **prediction}


def run_pipeline(args: argparse.Namespace) -> List[Dict[str, str]]:
    load_dotenv()
    claims = read_csv_rows(args.claims)
    if args.limit and args.limit > 0:
        claims = claims[: args.limit]
    history_by_user = load_history(args.history)
    cache_dir = args.cache_dir or (".cache/groq_claims" if args.provider == "groq" else ".cache/gemini_claims")
    if args.provider == "groq":
        from groq_client import GroqClient

        client = GroqClient(cache_dir=cache_dir, use_cache=not args.no_cache)
    else:
        from gemini_client import GeminiClient

        client = GeminiClient(cache_dir=cache_dir, use_cache=not args.no_cache)

    rows: List[Dict[str, str]] = []
    for index, claim in enumerate(claims, start=1):
        row = predict_claim(
            claim=claim,
            history_by_user=history_by_user,
            requirements_path=args.requirements,
            image_root=args.image_root,
            client=client,
            strategy=args.strategy,
        )
        rows.append(row)
        print(json.dumps({"row": index, "user_id": claim.get("user_id"), "status": row["claim_status"]}, ensure_ascii=False))

    write_csv_rows(args.output, rows, REQUIRED_COLUMNS)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
