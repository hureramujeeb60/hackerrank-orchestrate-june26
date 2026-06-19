# Evaluation Report

## Strategies Compared

### Strategy A: Basic Vision Prompt

This baseline sends the claim text and images to a vision model and asks for the required JSON fields. It is simple and low-overhead, but it does not explicitly ground the model in the evidence checklist, user history constraints, allowed enum values, or consistency rules. Expected failure modes are invalid labels, overuse of user history when deciding claim status, and weaker handling of multi-image identity mismatches.

### Strategy B: Structured Groq Vision Reviewer Prompt plus Validation

The selected strategy sends claim text, image IDs, user history, relevant evidence requirements, and the allowed labels to Groq's `meta-llama/llama-4-scout-17b-16e-instruct` vision model. The prompt requires JSON only and states the visual decision rules: images are primary, user history only affects risk context, absent or unverifiable claimed parts produce `not_enough_information`, visible claimed damage produces `supported`, and visible absence of claimed damage produces `contradicted`. A deterministic validation layer normalizes labels, fills missing fields, and prevents impossible combinations such as `claim_status=supported` with `evidence_standard_met=false`.

Strategy B was selected because the competition requires exact CSV schema compliance and reproducible behavior. The model still makes the visual judgment, while the validation layer keeps the output evaluable.

## Local Sample Evaluation

Run:

```bash
python code/evaluation/evaluate_sample.py
```

The script runs `code/main.py` on `dataset/sample_claims.csv`, writes `code/evaluation/sample_predictions.csv`, and reports field-level accuracy for:

- `claim_status`
- `issue_type`
- `object_part`
- `evidence_standard_met`
- `valid_image`
- `severity`

If `GROQ_API_KEY` is not configured, the pipeline emits conservative fallback rows so schema and runtime behavior can still be tested. Those fallback rows are not intended to represent final visual accuracy.

## Operational Analysis

The sample set has 20 claims and the test set has 44 claims. The pipeline makes one Groq call per uncached claim, so a full fresh run is approximately 20 sample calls and 44 test calls. The test images contain about 80 image files. Large images are compressed when Pillow is installed and a file is above the configured byte threshold.

Approximate usage depends on image resolution and Groq accounting. The text prompt is intentionally concise, typically under a few thousand tokens per claim, with one to three images attached. The JSON response is small, usually under 300 output tokens. With caching enabled in `code/.cache/groq`, repeated runs should make zero additional model calls for unchanged claims, prompts, model, and image bytes.

Runtime is dominated by model latency and rate limits. At a conservative 1 to 5 seconds per uncached claim, the 44-row test set should take roughly 1 to 4 minutes, plus retry time for transient errors. The client uses exponential backoff and local caching to reduce repeated cost. Pricing should be calculated from the current Groq model rates at run time; no API keys or billing data are stored in code, logs, cache, README, or output.
