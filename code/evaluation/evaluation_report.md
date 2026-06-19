# Evaluation Report

## Dataset

The evaluation workflow uses `dataset/sample_claims.csv`, which contains 20 labeled examples across cars, laptops, and packages. The final pipeline writes predictions for `dataset/claims.csv`, which contains 44 unlabeled claims.

## Strategies Compared

Strategy A is a no-model keyword heuristic baseline. It reads only the claim transcript and object type, extracts likely issue and part keywords, and assumes clear textual claims are supported. It is intentionally weak because it does not inspect images, but it provides a reproducible comparison point without spending API quota.

Strategy B is the selected final strategy. It uses Groq with `meta-llama/llama-4-scout-17b-16e-instruct` to produce per-image visual observations, then applies deterministic adjudication rules for claim status, evidence standard, risk flags, issue type, object part, supporting image IDs, validity, and severity.

Strategy B was selected because this task has a strict schema and several decision traps: user history should add risk but not decide the claim, absent damage should be contradicted only when the relevant part is visible, conflicting image identity should become `not_enough_information`, and unusable images should not be over-supported. The observation prompt plus deterministic adjudicator is more reproducible than asking the model to make every business decision directly.

## Metrics

Run:

```bash
python code/evaluation/evaluate_sample.py --strategy offline_comparison --predictions-input code/evaluation/sample_predictions.csv
```

The script reports exact-match accuracy for `claim_status`, `issue_type`, `object_part`, `evidence_standard_met`, `severity`, and `valid_image`, plus a claim-status confusion summary and row-level mismatches. The command above is score-only for the final structured predictions, so it does not call Groq or consume rate limits.

Latest sample comparison:

| Metric | Heuristic baseline | Final structured strategy |
|---|---:|---:|
| Rows expected | 20 | 20 |
| Rows predicted | 20 | 20 |
| Claim status accuracy | 0.60 | 1.00 |
| Issue type accuracy | 0.45 | 0.70 |
| Object part accuracy | 0.60 | 0.85 |
| Evidence standard accuracy | 0.80 | 1.00 |
| Severity accuracy | 0.45 | 0.40 |
| Valid image accuracy | 0.90 | 0.95 |

The final strategy is used for `output.csv`. The latest final output has 44 rows, matching `dataset/claims.csv`, and uses the exact required column order.

## Operational Analysis

Approximate model calls:

- Sample set: 20 Groq calls to generate the saved final structured sample predictions. The offline comparison command uses 0 calls because it scores `code/evaluation/sample_predictions.csv`.
- Test set: 44 calls for the final structured strategy.

Images processed:

- The sample and test CSVs reference all local submitted images for each claim. Multi-image rows are sent in a single model request per claim.

Token and cost assumptions:

- Prompt text is roughly 900-1,600 tokens per claim depending on history and requirements.
- Output is roughly 150-300 tokens per claim.
- Image token usage depends on Groq's model accounting and image resolution.
- Approximate full-test usage is 44 multimodal requests. Use current Groq pricing for the selected vision model because pricing can change.

Latency and rate limits:

- The runner processes claims sequentially to keep RPM/TPM behavior predictable.
- Responses are cached under `.cache/groq_claims` or the configured cache directory, so repeated evaluation or output generation does not call Groq again for unchanged claim/image/prompt inputs.
- The sample evaluator supports `--predictions-input` to score existing predictions without model calls, and refuses mismatched rows by default to avoid comparing the 44-row test output against the 20-row labeled sample set.
- The client retries transient or malformed responses and falls back to a valid conservative row if the API key, SDK, or response is unavailable.

## Reproducibility

The code reads the Groq key only from `GROQ_API_KEY`, uses `GROQ_MODEL` with default `meta-llama/llama-4-scout-17b-16e-instruct`, writes exactly the required output columns, and does not hardcode labels for specific claim IDs or image files.
