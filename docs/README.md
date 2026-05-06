# EmpathRAG Documentation Index

This folder is organized by purpose so the project does not turn into a pile of loose notes.

## Start Here

- [Master checklist](planning/MASTER_CHECKLIST.md): current single source of truth for the EmpathRAG Core sprint.
- [Current status audit](planning/CURRENT_STATUS_AUDIT.md): what exists, what works, and what remains risky.
- [Karthik Core Dataset V2 audit](audits/KARTHIK_CORE_DATASET_V2_AUDIT.md): dataset ingest result, Eval A/B metrics, and remaining data caveats.
- [Full Opus catch-up audit](audits/OPUS_FULL_PROJECT_AUDIT_2026_05_06.md): end-to-end project state, MVP gaps, metrics, and next product/research decisions.
- [Core architecture](architecture/EMPATHRAG_CORE_ARCHITECTURE.md): runtime design and pipeline structure.
- [Paper framing](research/PAPER_FRAMING.md): research story, claims, baselines, and evaluation framing.

## Demo

- [MSML class demo script](demo/MSML_CLASS_DEMO_SCRIPT.md)
- [Core demo script](demo/CORE_DEMO_SCRIPT.md)
- [Demo readiness checklist](demo/DEMO_READINESS_CHECKLIST.md)

## Research And Planning

- [Safety and dataset plan](research/SAFETY_AND_DATASET_PLAN.md)
- [Project memory handoff](planning/PROJECT_MEMORY_HANDOFF.md)

## Team Handoffs

Karthik/data tasks live under [team/karthik](team/karthik/):

- [Core dataset V2 request](team/karthik/CORE_DATASET_V2_REQUEST.md)
- [Send-now dataset request](team/karthik/send_now/EMPATHRAG_CORE_DATASET_REQUEST_SEND_NOW.md)
- [Eval dataset task](team/karthik/EVAL_DATASET_TASK.md)
- [V2 corpus audit](team/karthik/V2_CORPUS_AUDIT.md)
- [Corpus cleanup request](team/karthik/CORPUS_CLEANUP_REQUEST.md)
- [Corpus integration steps](team/karthik/CORPUS_INTEGRATION_STEPS.md)
- [Curated corpus handoff](team/karthik/CURATED_CORPUS_HANDOFF.md)
- [Curated corpus handoff PDF](team/karthik/CURATED_CORPUS_HANDOFF.pdf)

## Naming Rules Going Forward

- Use `planning/` for live task lists, audits, and handoff memory.
- Use `research/` for paper framing, evaluation plans, and safety claims.
- Use `demo/` for presentation scripts and demo readiness.
- Use `team/karthik/` for teammate-facing data instructions and audits.
- Avoid adding new loose Markdown files directly under `docs/`.
