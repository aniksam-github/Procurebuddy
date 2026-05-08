# Testing Harness

This folder holds fresh evaluation runs outside `python-ai-service`.

## Fresh smoke run

```powershell
python testing/run_fresh_eval.py
```

## Fresh smoke run with strict evaluator

```powershell
python testing/run_fresh_eval.py --strict-eval
```

This mode now uses a balanced contract:
- `semantic >= 0.6`
- `relevance >= 0.6`
- `FINAL DECISION:` must be present

That creates a timestamped folder under `testing/runs/` with:

- `eval_report.csv`
- `eval_failed_cases.json`
- `summary.json`

## Custom run

```powershell
python testing/run_fresh_eval.py --dataset tests/eval_framework/test_cases.json --limit 50 --label full
```

## Custom strict run

```powershell
python testing/run_fresh_eval.py --dataset tests/eval_framework/test_cases.json --limit 50 --label full_strict --strict-eval
```

## Lower-load balanced run

```powershell
python testing/run_fresh_eval.py --dataset tests/eval_framework/test_cases.json --limit 20 --batch-size 3 --timeout 180 --strict-eval
```

## Summarize an existing report

```powershell
python testing/summarize_eval.py testing/runs/<timestamp>_<label>/eval_report.csv
```
