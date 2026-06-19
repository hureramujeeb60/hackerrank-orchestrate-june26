from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from schema import REQUIRED_COLUMNS  # noqa: E402


FIELDS_TO_SCORE = [
    "claim_status",
    "issue_type",
    "object_part",
    "evidence_standard_met",
    "severity",
    "valid_image",
]

PREDICTION_DEFAULTS = {
    "evidence_standard_met": "false",
    "evidence_standard_met_reason": "Keyword baseline did not inspect images.",
    "risk_flags": "manual_review_required",
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "Keyword baseline lacks visual evidence.",
    "supporting_image_ids": "none",
    "valid_image": "true",
    "severity": "unknown",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(expected: List[Dict[str, str]], predicted: List[Dict[str, str]]) -> Dict[str, object]:
    metrics: Dict[str, object] = {"rows_expected": len(expected), "rows_predicted": len(predicted)}
    for field in FIELDS_TO_SCORE:
        total = min(len(expected), len(predicted))
        correct = sum(1 for exp, pred in zip(expected, predicted) if exp.get(field) == pred.get(field))
        metrics[f"{field}_accuracy"] = round(correct / total, 4) if total else 0.0
    status_pairs = Counter((exp.get("claim_status"), pred.get("claim_status")) for exp, pred in zip(expected, predicted))
    metrics["claim_status_confusion"] = {f"{a}->{b}": n for (a, b), n in sorted(status_pairs.items())}
    metrics["row_mismatches"] = build_row_mismatches(expected, predicted)
    return metrics


def build_row_mismatches(expected: List[Dict[str, str]], predicted: List[Dict[str, str]]) -> List[Dict[str, object]]:
    mismatches: List[Dict[str, object]] = []
    for index, (exp, pred) in enumerate(zip(expected, predicted), start=1):
        fields = [field for field in FIELDS_TO_SCORE if exp.get(field) != pred.get(field)]
        if not fields:
            continue
        mismatches.append(
            {
                "row": index,
                "user_id": exp.get("user_id", ""),
                "fields": fields,
                "expected_status": exp.get("claim_status", ""),
                "predicted_status": pred.get("claim_status", ""),
            }
        )
    return mismatches


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the claim verifier on sample_claims.csv.")
    parser.add_argument("--sample", default=str(ROOT / "dataset/sample_claims.csv"))
    parser.add_argument("--history", default=str(ROOT / "dataset/user_history.csv"))
    parser.add_argument("--requirements", default=str(ROOT / "dataset/evidence_requirements.csv"))
    parser.add_argument("--image-root", default=str(ROOT / "dataset"))
    parser.add_argument(
        "--strategy",
        choices=["heuristic", "basic", "structured", "both", "offline_comparison"],
        default="structured",
    )
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache/groq_claims"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="Disable cache when generating sample predictions")
    parser.add_argument("--predictions-input", default="", help="Score an existing predictions CSV without model calls")
    parser.add_argument("--allow-row-mismatch", action="store_true", help="Allow score-only comparison when input rows differ")
    parser.add_argument("--predictions-output", default="", help="Optional path to save sample predictions")
    return parser


def run_strategy(args: argparse.Namespace, strategy: str) -> Dict[str, object]:
    expected = read_rows(Path(args.sample))
    if args.limit:
        expected = expected[: args.limit]
    if strategy == "heuristic":
        predicted = [heuristic_prediction(row) for row in expected]
        metrics = score(expected, predicted)
        metrics["strategy"] = strategy
        metrics["missing_required_columns"] = []
        metrics["predictions_output"] = ""
        return metrics

    if args.predictions_input:
        predicted = read_rows(Path(args.predictions_input))
        if args.limit:
            predicted = predicted[: args.limit]
        assert_matching_inputs(expected, predicted, allow_mismatch=args.allow_row_mismatch)
        missing_cols = [col for col in REQUIRED_COLUMNS if predicted and col not in predicted[0]]
        metrics = score(expected, predicted)
        metrics["strategy"] = f"{strategy}_score_only"
        metrics["missing_required_columns"] = missing_cols
        metrics["predictions_output"] = args.predictions_input
        return metrics

    with tempfile.TemporaryDirectory() as tmp:
        if args.predictions_output and args.strategy == "both":
            base = Path(args.predictions_output)
            output = base.with_name(f"{base.stem}_{strategy}{base.suffix or '.csv'}")
        elif args.predictions_output:
            output = Path(args.predictions_output)
        else:
            output = Path(tmp) / f"sample_{strategy}.csv"
        if args.predictions_output:
            output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(CODE_DIR / "main.py"),
            "--claims",
            args.sample,
            "--history",
            args.history,
            "--requirements",
            args.requirements,
            "--image-root",
            args.image_root,
            "--output",
            str(output),
            "--cache-dir",
            args.cache_dir,
            "--strategy",
            strategy,
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.no_cache:
            cmd.append("--no-cache")
        subprocess.run(cmd, cwd=ROOT, check=True)
        predicted = read_rows(output)
    missing_cols = [col for col in REQUIRED_COLUMNS if predicted and col not in predicted[0]]
    metrics = score(expected, predicted)
    metrics["strategy"] = strategy
    metrics["missing_required_columns"] = missing_cols
    metrics["predictions_output"] = str(output) if args.predictions_output else ""
    return metrics


def heuristic_prediction(row: Dict[str, str]) -> Dict[str, str]:
    claim = row.get("user_claim", "").lower()
    obj = row.get("claim_object", "").lower()
    pred = dict(PREDICTION_DEFAULTS)
    pred["valid_image"] = "true" if row.get("image_paths") else "false"

    part = "unknown"
    for candidate, phrases in part_keywords(obj):
        if any(phrase in claim for phrase in phrases):
            part = candidate
            break

    issue = "unknown"
    for candidate, phrases in issue_keywords(obj):
        if any(phrase in claim for phrase in phrases):
            issue = candidate
            break

    if issue != "unknown" and part != "unknown":
        pred.update(
            {
                "evidence_standard_met": "true",
                "evidence_standard_met_reason": "Keyword baseline found a claim issue and part in the transcript.",
                "risk_flags": "none",
                "issue_type": issue,
                "object_part": part,
                "claim_status": "supported",
                "claim_status_justification": "Keyword baseline assumes the stated claim is supported.",
                "supporting_image_ids": first_image_id(row.get("image_paths", "")),
                "severity": "medium" if issue not in {"scratch", "dent"} else "low",
            }
        )

    return {**row, **pred}


def part_keywords(obj: str) -> List[tuple[str, tuple[str, ...]]]:
    if obj == "car":
        return [
            ("rear_bumper", ("rear bumper", "back bumper")),
            ("front_bumper", ("front bumper", "front side")),
            ("windshield", ("windshield", "front glass")),
            ("side_mirror", ("side mirror", "mirror")),
            ("headlight", ("headlight",)),
            ("hood", ("hood", "top panel")),
            ("door", ("door", "door panel")),
        ]
    if obj == "laptop":
        return [
            ("hinge", ("hinge",)),
            ("trackpad", ("trackpad",)),
            ("keyboard", ("keyboard", "keys")),
            ("screen", ("screen", "display")),
            ("corner", ("corner",)),
        ]
    if obj == "package":
        return [
            ("contents", ("contents", "item", "product")),
            ("seal", ("seal", "tape", "opened")),
            ("package_corner", ("corner",)),
            ("package_side", ("side", "surface", "outside")),
            ("box", ("box", "package")),
        ]
    return []


def issue_keywords(obj: str) -> List[tuple[str, tuple[str, ...]]]:
    common = [
        ("broken_part", ("broken", "damaged")),
        ("crack", ("crack", "cracked", "shattered")),
        ("dent", ("dent", "bump")),
        ("scratch", ("scratch", "scrape", "mark")),
    ]
    if obj == "package":
        return [
            ("missing_part", ("missing", "not inside")),
            ("crushed_packaging", ("crushed",)),
            ("torn_packaging", ("torn", "opened", "seal")),
            ("water_damage", ("water", "wet")),
            ("stain", ("stain",)),
        ]
    if obj == "laptop":
        return [("stain", ("stain", "sticky", "water")), *common]
    return common


def first_image_id(paths: str) -> str:
    first = (paths or "").split(";")[0].strip()
    if not first:
        return "none"
    return Path(first).stem or "none"


def assert_matching_inputs(expected: List[Dict[str, str]], predicted: List[Dict[str, str]], allow_mismatch: bool) -> None:
    if allow_mismatch:
        return
    key_cols = ["user_id", "image_paths", "user_claim", "claim_object"]
    if len(expected) != len(predicted):
        raise SystemExit(
            f"Predictions row count ({len(predicted)}) does not match sample row count ({len(expected)}). "
            "Use predictions generated from dataset/sample_claims.csv, or pass --allow-row-mismatch only for debugging."
        )
    for index, (exp, pred) in enumerate(zip(expected, predicted), start=1):
        if any(exp.get(col, "") != pred.get(col, "") for col in key_cols):
            raise SystemExit(
                f"Predictions input row {index} does not match dataset/sample_claims.csv. "
                "Use code/evaluation/sample_predictions.csv or regenerate sample predictions from dataset/sample_claims.csv."
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.strategy == "both":
        strategies = ["basic", "structured"]
    elif args.strategy == "offline_comparison":
        strategies = ["heuristic", "structured"]
    else:
        strategies = [args.strategy]
    for strategy in strategies:
        metrics = run_strategy(args, strategy)
        print(f"\nStrategy: {strategy}")
        for key, value in metrics.items():
            if key != "strategy":
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
