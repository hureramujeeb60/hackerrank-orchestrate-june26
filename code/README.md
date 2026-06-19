# HackerRank Orchestrate Claim Verification Pipeline

This folder contains a command-line Python solution for the multi-modal evidence review challenge. It reads the provided CSV files, sends each claim and its local images to Groq's vision API, validates the JSON response, caches model outputs, and writes the required `output.csv` schema.

## Setup

Use Python 3.10+.

```bash
pip install -r code/requirements.txt
```

Set one API key environment variable:

PowerShell:

```powershell
$env:GROQ_API_KEY="your-key"
```

macOS/Linux:

```bash
export GROQ_API_KEY="your-key"
```

The model defaults to `meta-llama/llama-4-scout-17b-16e-instruct`; override it with `GROQ_MODEL` or `--model`.

## Run Final Predictions

```bash
python code/main.py --claims dataset/claims.csv --history dataset/user_history.csv --requirements dataset/evidence_requirements.csv --image-root dataset --output output.csv
```

For rate-limited keys, add a small delay to avoid request quota bursts:

```bash
python code/main.py --claims dataset/claims.csv --history dataset/user_history.csv --requirements dataset/evidence_requirements.csv --image-root dataset --output output.csv --sleep-between-calls 4 --no-fallback
```

The output file has exactly the required columns:

```text
user_id,image_paths,user_claim,claim_object,evidence_standard_met,evidence_standard_met_reason,risk_flags,issue_type,object_part,claim_status,claim_status_justification,supporting_image_ids,valid_image,severity
```

## Evaluate on Sample Data

```bash
python code/evaluation/evaluate_sample.py
```

This writes `code/evaluation/sample_predictions.csv` and prints field-level metrics against `dataset/sample_claims.csv`.

## Caching and Retries

Groq responses are cached under `code/.cache/groq` by default. The cache key includes the claim row, selected history, selected evidence requirements, model, prompt version, and image bytes. Rerunning the same command reuses cached responses.

The client retries transient failures with exponential backoff. If Groq is unavailable and fallback is enabled, the pipeline emits conservative `not_enough_information` rows so the CSV contract can still be tested. For a strict competition run, configure the API key and use `--no-fallback` if you want the command to fail instead of writing fallback rows.

Fallback rows are not cached as final predictions. Successful Groq responses are cached, so reruns continue from completed claims.

## Files

- `main.py`: command-line entry point
- `prompts.py`: structured reviewer prompt
- `groq_client.py`: Groq SDK integration, local image data URLs, JSON parsing, retries
- `schema.py`: allowed labels, normalization, validation, conservative fallback
- `utils.py`: CSV, image loading/compression, caching helpers
- `evaluation/evaluate_sample.py`: sample evaluation workflow
- `evaluation/evaluation_report.md`: strategy comparison and operational analysis
