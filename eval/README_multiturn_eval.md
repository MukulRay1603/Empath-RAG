# Multi-Turn Evaluation Harness

This harness evaluates the V2.5 fast backend without loading the full local LLM stack.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe eval\run_multiturn_eval.py
```

Inputs:

- `eval/multiturn_scenarios.jsonl`

Default output:

- `eval/multiturn_results.json`

Metrics:

- route accuracy
- safety tier accuracy
- missed escalation count
- unsafe generation count
- pure-validation/no-action count
- ungrounded action count
- average latency

This is a deterministic project harness, not a clinical validation.
