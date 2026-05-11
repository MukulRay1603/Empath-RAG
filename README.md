# EmpathRAG

<div align="center">

**A guarded conversational RAG support navigator for UMD students.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![University](https://img.shields.io/badge/UMD-MSML641-E03A3E?style=flat-square)](https://umd.edu)

*A research prototype, not a clinical product.*

</div>

---

EmpathRAG listens to a student's message, decides what kind of support is needed, retrieves grounded UMD resources, and produces a single practical next step. Crisis content is intercepted before any language model is invoked. The system is **not** therapy, diagnosis, counseling, or an emergency service.

Most chatbots route every message through the language model. This one doesn't. A deterministic safety planner decides *what* to say; the LLM only paraphrases *how*.

## Headline result

Same underlying model (Llama 3.3 70B via Groq), 28 escalation scenarios from a multi-turn safety benchmark:

| System | Missed escalation rate | Harm endorsement |
|---|---:|---:|
| **EmpathRAG Core (full stack)** | **0 / 28 (0.0%)** | **0** |
| Unguarded Llama 3.3 70B (no pipeline) | 9 / 28 (32.1%) | 2 turns |

Non-overlapping 95% CIs. The entire delta is architectural. Full evaluation results and reproducibility commands are in [`docs/research/PAPER_FRAMING.md`](docs/research/PAPER_FRAMING.md) and [`docs/research/REPRODUCIBILITY.md`](docs/research/REPRODUCIBILITY.md).

## Architecture

```
Student message
     │
     ├──▶ [1] Stage-1 lexical safety precheck       crisis intercept · LLM bypassed
     ├──▶ [2] Optional DeBERTa NLI guardrail        off in live demo for latency
     ├──▶ [3] Hybrid route + tier classifier        14 routes × 4 safety tiers
     ├──▶ [4] Resource registry filter              34 verified UMD/national entries
     ├──▶ [5] Stage-aware planner                   LISTEN → PERMISSION → OFFER → CLARIFY
     ├──▶ [6] Plan-and-rephrase                     Groq → Anthropic → fallback
     │                                              + post-rephrase trust boundary
     └──▶ [7] Output guard                          missing-action / dependency /
                                                    sycophancy / fabrication
```

A per-layer ablation evaluation quantifies each layer's contribution to the safety floor. See [`docs/research/PAPER_FRAMING.md`](docs/research/PAPER_FRAMING.md).

## Quickstart

```powershell
# 1. Clone + venv
git clone https://github.com/MukulRay1603/Empath-RAG.git
cd Empath-RAG
python -m venv venv
.\venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. .env at repo root
#    GROQ_API_KEY=gsk_...
#    ANTHROPIC_API_KEY=sk-ant-...   (optional fallback)

# 4. Launch the Gradio demo
$env:EMPATHRAG_DEMO_BACKEND='fast'
$env:EMPATHRAG_REPHRASER_ENABLED='1'
.\venv\Scripts\python.exe -u demo\app.py

# 5. Open http://127.0.0.1:7860/
```

Without API keys the system runs in deterministic-template mode — all safety layers still function; only the natural-language paraphrasing is offline.

## Documentation

| Doc | Audience | What's in it |
|---|---|---|
| [`docs/architecture/EMPATHRAG_CORE_ARCHITECTURE.md`](docs/architecture/EMPATHRAG_CORE_ARCHITECTURE.md) | Engineers, reviewers | Runtime design, pipeline order, the seven safety layers |
| [`docs/research/PAPER_FRAMING.md`](docs/research/PAPER_FRAMING.md) | Reviewers, the paper | Research story, baselines, full evaluation results, V1 motivation |
| [`docs/research/REPRODUCIBILITY.md`](docs/research/REPRODUCIBILITY.md) | Reviewers, forks | Exact commands per evaluation, expected results, what's tracked vs untracked |
| [`docs/research/ERROR_ANALYSIS.md`](docs/research/ERROR_ANALYSIS.md) | Reviewers | Seven categories of observed failure modes with mitigations and residual risk |
| [`docs/research/PRIVACY_AND_DATA_FLOW.md`](docs/research/PRIVACY_AND_DATA_FLOW.md) | Students, clinicians | What data goes where, retention, how to clear local state |
| [`docs/research/HIPAA_FERPA_GAP_ANALYSIS.md`](docs/research/HIPAA_FERPA_GAP_ANALYSIS.md) | Compliance reviewers | Explicit accounting of deployment-readiness gaps |

## Repo

```
src/pipeline/      core.py · rephraser.py · response_planner.py · safety_policy.py
                   output_guard.py · ml_router.py · service_graph.py · llm_safety.py
                   support_plan.py · voice.py · v2_schema.py
demo/              app.py            Gradio UI + streaming + safety-pipeline viz
eval/              run_multiturn_eval.py · run_ablation_eval.py · run_unguarded_baseline.py
                   sweep_*.py (5 targeted failure-mode probe sweeps)
                   audit_resource_urls.py
data/curated/      service_graph.jsonl  (34 verified UMD/national service objects)
tests/             test_v25_support_navigator.py  (21 regression tests)
docs/              architecture/ · research/
```

Intentionally untracked (regenerable or sensitive): `data/curated/indexes/`, `models/router/`, `Data_Karthik/`, `.env`, generated eval reports.

## Contributors

- **Mukul Rayana** — UMD MSML, project lead.
- **Karthik** — UMD MSML, dataset and curated-corpus delivery.

This prototype is built openly for academic use. It is not a UMD product or service.

## License

[MIT](LICENSE). Dataset and third-party model licenses vary; see `docs/research/PAPER_FRAMING.md` for the full provenance.
