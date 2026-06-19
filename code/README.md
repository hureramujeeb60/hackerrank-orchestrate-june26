# Multimodal Claim Verification Pipeline

This folder contains a command-line Python pipeline for the HackerRank Orchestrate multimodal claim verification task. It reads the provided CSV files and local images, calls a Groq-hosted vision model for visual review, validates the model JSON, and writes `output.csv` in the required schema.

## Setup

Install dependencies:

```bash
python -m pip install -r code/requirements.txt
```

Set an API key:

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your-key-here"
```

macOS/Linux:

```bash
export GROQ_API_KEY="your-key-here"
```

You can also put the key in a local `.env` file at the repository root:

```text
GROQ_API_KEY=your-key-here
```

The default Groq model is `meta-llama/llama-4-scout-17b-16e-instruct`, selected because Groq documents it for vision/image understanding, JSON mode, and multimodal API usage. Override it with:

```bash
export GROQ_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
```

## Run Final Predictions

From the repository root:

```bash
python code/main.py --claims dataset/claims.csv --history dataset/user_history.csv --requirements dataset/evidence_requirements.csv --image-root dataset --output output.csv
```

The runner defaults to Groq. To use the older Gemini client instead, pass `--provider gemini` and configure `GEMINI_API_KEY`.

The output file contains exactly the required columns in order:

`user_id,image_paths,user_claim,claim_object,evidence_standard_met,evidence_standard_met_reason,risk_flags,issue_type,object_part,claim_status,claim_status_justification,supporting_image_ids,valid_image,severity`

## Evaluate On Sample Data

```bash
python code/evaluation/evaluate_sample.py --strategy offline_comparison --predictions-input code/evaluation/sample_predictions.csv
```

This compares a no-model heuristic baseline with the saved final structured sample predictions and prints exact-match metrics for important fields. It is score-only for the structured strategy and does not call Groq.

To save row-aligned sample predictions for inspection:

```bash
python code/evaluation/evaluate_sample.py --strategy structured --cache-dir .cache/groq_sample_live --predictions-output code/evaluation/sample_predictions.csv
```

To regenerate live sample predictions, run the command above with a valid `GROQ_API_KEY`. To evaluate an existing predictions file without API usage, pass `--predictions-input`; the evaluator verifies that the prediction rows match `dataset/sample_claims.csv` before scoring.

## Cache And Fallbacks

Groq responses are cached under `.cache/groq_claims` based on claim data, prompt strategy, model, and image metadata. Reruns reuse cached responses unless `--no-cache` is passed.

If the API key or SDK is missing, or the model returns malformed JSON after retries, the pipeline writes a conservative valid row with `claim_status=not_enough_information`. This keeps the command runnable for schema checks, but real competition predictions require a configured Groq API key.
