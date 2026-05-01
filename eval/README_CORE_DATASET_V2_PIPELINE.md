# EmpathRAG Core Dataset V2 Pipeline

Use this when Karthik sends `empathrag_core_dataset_v2/`.

## Expected Delivery Folder

Place the folder at:

```powershell
Data_Karthik\empathrag_core_dataset_v2
```

Required files:

- `README_dataset_notes.md`
- `single_turn_labeled.csv`
- `multi_turn_scenarios.jsonl`
- `source_target_map.csv`
- `risky_ambiguous_cases.csv`
- `resource_profile_additions.csv`

## Ingest And Validate

```powershell
.\venv\Scripts\python.exe -B eval\ingest_core_dataset_v2.py --delivery-dir Data_Karthik\empathrag_core_dataset_v2
```

Outputs:

- `eval\empathrag_core_supervised.csv`
- `eval\multiturn_scenarios.jsonl`
- `eval\core_dataset_v2_ingest_report.json`
- `eval\core_dataset_v2_ingest_report.md`

The script validates labels, required columns, duplicate IDs, multi-turn scenario
shape, and resource-profile additions. It does not automatically merge resource
additions into the runtime registry; those must be manually reviewed first.

## Train Router

```powershell
.\venv\Scripts\python.exe -B eval\train_ml_router.py
```

This writes local model artifacts under `models\router\`, which is ignored by
git. If models are missing, the demo still falls back to deterministic routing.

## Eval A: Single-Turn Router Ablation

```powershell
.\venv\Scripts\python.exe -B eval\run_router_eval.py
.\venv\Scripts\python.exe -B eval\run_empathrag_core_eval.py
```

Primary metric: route accuracy.

Secondary metrics: safety tier accuracy, intercept accuracy, source hit rate,
source avoid-list violations, unsafe generation, no-action responses,
ungrounded action, and latency.

## Eval B: Multi-Turn Safety Benchmark

```powershell
.\venv\Scripts\python.exe -B eval\run_multiturn_eval.py
```

Primary metric: missed escalation rate.

Secondary metrics: dependency reinforcement, pure validation/no-action,
unsafe generation, method leakage, and final safety tier correctness.

## Smoke Test With Fixture

```powershell
.\venv\Scripts\python.exe -B eval\ingest_core_dataset_v2.py `
  --delivery-dir eval\fixtures\core_dataset_v2_sample `
  --output eval\empathrag_core_supervised.sample.csv `
  --multiturn-output eval\multiturn_scenarios.sample.jsonl `
  --report-json eval\core_dataset_v2_ingest_report.sample.json `
  --report-md eval\core_dataset_v2_ingest_report.sample.md
```

This fixture is only for testing the ingest gate; it is not research data.
