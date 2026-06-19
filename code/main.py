from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

from groq_client import GroqClient, GroqUnavailable
from prompts import PROMPT_VERSION, build_reviewer_prompt
from schema import OUTPUT_COLUMNS, fallback_prediction, normalize_prediction
from utils import (
    index_by,
    load_images,
    load_json,
    read_csv,
    save_json,
    select_requirements,
    split_image_paths,
    stable_claim_key,
    write_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HackerRank Orchestrate claim verifier.")
    parser.add_argument("--claims", default="dataset/claims.csv", help="Claims CSV path.")
    parser.add_argument("--history", default="dataset/user_history.csv", help="User history CSV path.")
    parser.add_argument(
        "--requirements",
        default="dataset/evidence_requirements.csv",
        help="Evidence requirements CSV path.",
    )
    parser.add_argument("--image-root", default="dataset", help="Root directory for relative image paths.")
    parser.add_argument("--output", default="output.csv", help="Prediction CSV path.")
    parser.add_argument("--cache-dir", default="code/.cache/groq", help="Local Groq response cache.")
    parser.add_argument("--model", default=None, help="Override GROQ_MODEL.")
    parser.add_argument("--max-retries", type=int, default=3, help="Groq retry attempts per uncached claim.")
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Seconds to sleep after each uncached successful Groq call; useful for RPM limits.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of emitting conservative rows when Groq is unavailable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    return parser


def predict_claim(
    claim: dict[str, str],
    history_by_user: dict[str, dict[str, str]],
    requirements_rows: list[dict[str, str]],
    image_root: str | Path,
    cache_dir: Path,
    client: GroqClient,
    allow_fallback: bool,
    sleep_between_calls: float = 0.0,
) -> dict[str, str]:
    image_path_values = split_image_paths(claim.get("image_paths"))
    images, missing_images = load_images(image_path_values, image_root)
    image_ids = [image.image_id for image in images]
    history = history_by_user.get(claim.get("user_id", ""), {})
    requirements = select_requirements(claim, requirements_rows)
    cache_key = stable_claim_key(claim, history, requirements, images, client.model, PROMPT_VERSION)
    cache_path = cache_dir / f"{cache_key}.json"
    cached = load_json(cache_path)

    if cached and not cached.get("fallback") and isinstance(cached.get("parsed"), dict):
        prediction = cached["parsed"]
    else:
        try:
            if missing_images and not images:
                raise GroqUnavailable(f"No loadable images; missing: {', '.join(missing_images)}")
            prompt = build_reviewer_prompt(claim, history, requirements, images)
            result = client.generate_json(prompt, images)
            prediction = result.parsed
            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)
            save_json(
                cache_path,
                {
                    "model": result.model,
                    "attempts": result.attempts,
                    "raw_text": result.raw_text,
                    "parsed": result.parsed,
                },
            )
        except GroqUnavailable as exc:
            if not allow_fallback:
                raise
            prediction = fallback_prediction(
                claim,
                image_ids,
                missing_images,
                reason=str(exc),
            )

    return normalize_prediction(claim, prediction, image_ids, history)


def run(args: argparse.Namespace) -> int:
    claims = read_csv(args.claims)
    if args.limit is not None:
        claims = claims[: args.limit]
    history_by_user = index_by(read_csv(args.history), "user_id")
    requirements = read_csv(args.requirements)
    client = GroqClient(model=args.model, max_retries=args.max_retries)
    cache_dir = Path(args.cache_dir)
    rows: list[dict[str, str]] = []

    for index, claim in enumerate(claims, start=1):
        row = predict_claim(
            claim=claim,
            history_by_user=history_by_user,
            requirements_rows=requirements,
            image_root=args.image_root,
            cache_dir=cache_dir,
            client=client,
            allow_fallback=not args.no_fallback,
            sleep_between_calls=args.sleep_between_calls,
        )
        rows.append(row)
        print(
            f"[{index}/{len(claims)}] {claim.get('user_id', '')} "
            f"{row['claim_status']} {row['issue_type']} {row['object_part']}",
            file=sys.stderr,
        )

    write_csv(args.output, rows, OUTPUT_COLUMNS)
    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    return 0


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
