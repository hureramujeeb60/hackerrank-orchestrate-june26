from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path


FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def field_accuracy(expected: list[dict[str, str]], predicted: list[dict[str, str]], field: str) -> tuple[int, int]:
    correct = 0
    total = min(len(expected), len(predicted))
    for exp, pred in zip(expected, predicted):
        if exp.get(field, "").strip() == pred.get(field, "").strip():
            correct += 1
    return correct, total


def print_confusion(expected: list[dict[str, str]], predicted: list[dict[str, str]], field: str) -> None:
    pairs = Counter((exp.get(field, ""), pred.get(field, "")) for exp, pred in zip(expected, predicted))
    print(f"\n{field} confusion (expected -> predicted):")
    for (exp, pred), count in pairs.most_common():
        print(f"  {exp or '<blank>'} -> {pred or '<blank>'}: {count}")


def run_pipeline(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "code/main.py",
        "--claims",
        args.sample,
        "--history",
        args.history,
        "--requirements",
        args.requirements,
        "--image-root",
        args.image_root,
        "--output",
        args.predictions,
        "--cache-dir",
        args.cache_dir,
    ]
    if args.no_fallback:
        command.append("--no-fallback")
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate predictions on dataset/sample_claims.csv.")
    parser.add_argument("--sample", default="dataset/sample_claims.csv")
    parser.add_argument("--history", default="dataset/user_history.csv")
    parser.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    parser.add_argument("--image-root", default="dataset")
    parser.add_argument("--predictions", default="code/evaluation/sample_predictions.csv")
    parser.add_argument("--cache-dir", default="code/.cache/groq_sample")
    parser.add_argument("--skip-run", action="store_true", help="Compare an existing predictions CSV.")
    parser.add_argument("--no-fallback", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.skip_run:
        run_pipeline(args)

    expected = read_rows(args.sample)
    predicted = read_rows(args.predictions)
    print(f"Expected rows: {len(expected)}")
    print(f"Predicted rows: {len(predicted)}")
    if len(expected) != len(predicted):
        print("Row count mismatch", file=sys.stderr)

    print("\nField accuracy:")
    for field in FIELDS:
        correct, total = field_accuracy(expected, predicted, field)
        pct = (correct / total * 100) if total else 0.0
        print(f"  {field}: {correct}/{total} ({pct:.1f}%)")

    print_confusion(expected, predicted, "claim_status")
    print_confusion(expected, predicted, "issue_type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
