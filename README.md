# EmpathRAG

<div align="center">

**Emotion-Aware Retrieval-Augmented Generation with Safety Guardrails for Student Mental Health**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()
[![University](https://img.shields.io/badge/UMD-MSML641-E03A3E?style=flat-square)](https://umd.edu)

*MSML641 · Applied Machine Learning · University of Maryland · Spring 2025*

</div>

---

## What Is This?

Standard RAG systems treat every message identically — as a neutral information request. A student in crisis and a student asking for study tips go through the same retrieval pipeline, with no emotional awareness and no safety gate.

**EmpathRAG fixes this.** It is a 5-stage NLP pipeline that classifies a student's emotional state *before* retrieval, rewrites queries based on that state, and intercepts crisis-level messages with a trained NLI classifier before the language model is ever invoked.

---

## Pipeline

```
Student Message
      │
      ├──▶ [1] Emotion Classifier     RoBERTa + LoRA  (CPU)
      │         distress · anxiety · frustration · neutral · hopeful
      │
      ├──▶ [2] Safety Guardrail       DeBERTa-v3 NLI  (CPU)
      │         ⚠ If crisis → return 988 lifeline · STOP · generator never runs
      │
      ├──▶ [3] Query Router           Deterministic templates + session tracker
      │         Rewrites query with emotion context · tracks 6 trajectory states
      │
      ├──▶ [4] FAISS Retrieval        all-mpnet-base-v2 · 1,674,369 vectors
      │         Emotion-match re-ranking · SQLite metadata sidecar
      │
      └──▶ [5] Mistral 7B Generator   Q4_K_M GGUF · 28/33 GPU layers
                Sliding 3-turn memory · 2-paragraph empathetic response
```

---

## Results

| Metric | Value | Target | |
|---|---|---|---|
| RoBERTa Emotion F1 (weighted) | **0.7127** | > 0.55 | ✅ |
| DeBERTa Crisis Recall | **0.9629** | > 0.80 | ✅ |
| DeBERTa Crisis Precision | **0.7951** | > 0.65 | ✅ |
| BERTScore F1 | **0.8266** | > 0.72 | ✅ |
| Wilcoxon p-value (Full vs BM25) | **3.62e-08** | < 0.05 | ✅ |
| Adversarial crisis recall (NLI) | **85%** | > 80% | ✅ |
| Euphemistic recall — NLI vs keyword | **100% vs 20%** | — | 🔑 |

### Ablation — Emotion Alignment Score

| Condition | Retrieval | Emotion Conditioning | Score |
|---|---|---|---|
| A — BM25 baseline | Sparse | None | 0.30 |
| C — Dense RAG | FAISS semantic | None | 0.50 |
| D — **Full EmpathRAG** | FAISS + emotion | Query rewrite + re-rank | **0.88** |

> Emotion conditioning contributes **+0.38** over pure dense retrieval (C→D). Wilcoxon signed-rank: p = 3.62e-08.

---

## Key Finding

The NLI-framed safety guardrail outperforms a keyword filter across every adversarial probe category that matters:

| Probe Category | EmpathRAG (NLI) | Keyword Filter |
|---|---|---|
| Direct crisis language | 100% | 80% |
| Euphemistic / indirect | **100%** | 20% |
| Negation bypass | **100%** | 60% |
| Bait-and-switch | 40% | 20% |

Keyword filters miss indirect phrasing. NLI understands semantic entailment. This is the core research finding.

---

## Demo

> 🔄 **Demo recording in progress** — Loom walkthrough coming soon.

The Gradio demo shows:
- Real-time emotion timeline with trajectory detection across turns
- Safety guardrail panel with Integrated Gradients token attribution highlights
- Multi-turn conversation memory (sliding 3-turn window)
- Session ID for human evaluation correlation

To run locally (requires models — see [Setup](#setup)):
```bash
python demo/app.py
```

---

## VRAM Budget (RTX 3060, 6 GB)

| Component | Device | VRAM |
|---|---|---|
| RoBERTa + DeBERTa | CPU | 0 MB |
| Sentence Transformer | GPU (encode only) | 440 MB |
| Mistral 7B Q4_K_M | GPU resident | 3,870 MB |
| KV cache + compute | GPU | ~630 MB |
| **Total peak** | | **~4,940 MB** ✅ |

---

## Repo Structure

```
Empath-RAG/
├── src/
│   ├── pipeline/
│   │   ├── pipeline.py          # 5-stage orchestrator · conversation memory
│   │   ├── query_router.py      # Emotion-conditioned query rewriting
│   │   └── session_tracker.py   # Trajectory detection (6 states)
│   ├── models/
│   │   ├── guardrail_ig.py      # DeBERTa NLI + Integrated Gradients
│   │   └── annotate_corpus.py   # Corpus emotion annotation (Colab A100)
│   └── data/
│       ├── preprocess.py        # GoEmotions 27→5 label collapse
│       ├── build_faiss_index.py # FAISS IVFFlat builder
│       └── build_nli_pairs.py   # NLI training pair construction
├── demo/
│   └── app.py                   # Gradio demo · emotion timeline · crisis panel
├── eval/
│   ├── run_ablation.py          # Conditions A / C / D
│   ├── run_bertscore.py         # BERTScore F1
│   ├── run_wilcoxon.py          # Wilcoxon signed-rank test
│   ├── run_adversarial.py       # NLI vs keyword filter · 30 probes
│   ├── test_prompts.json        # 50 prompts · 10 per emotion class
│   └── adversarial_probes.json  # 30 probes · 6 categories × 5
├── notebooks/
│   ├── colab_emotion_classifier.ipynb   # RoBERTa + LoRA · A100
│   └── colab_deberta_guardrail.ipynb    # DeBERTa NLI · A100 · bf16
└── data/
    └── MANIFEST.md              # Dataset download instructions
```

---

## Setup

**Requirements:** Python 3.12 · CUDA 12.1 · 6 GB VRAM · 16 GB RAM · ~15 GB disk

> ⚠️ Install order is critical — do not skip steps.

```bash
# 1. Clone
git clone https://github.com/MukulRay1603/Empath-RAG.git
cd Empath-RAG
python -m venv venv
venv\Scripts\activate        # Windows

# 2. PyTorch with CUDA — install FIRST
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. llama-cpp-python CUDA wheel — install SECOND
pip install "llama_cpp_python==0.3.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 4. Everything else
pip install -r requirements.txt

# 5. Force correct numpy (faiss requires < 2.0)
pip install "numpy==1.26.4" --force-reinstall
```

**Model downloads** — see [`data/MANIFEST.md`](data/MANIFEST.md) for dataset download instructions.

Mistral 7B GGUF → download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` from [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) → place at `models/generator/mistral-7b-instruct-v0.2.Q4_K_M.gguf`

---

## Datasets

| Dataset | Role | License |
|---|---|---|
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Emotion classifier training | Apache 2.0 |
| [Reddit Mental Health](https://zenodo.org/records/3941387) | FAISS retrieval corpus | CC BY 4.0 |
| [Suicide Detection](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) | Guardrail NLI training | Public |
| [Empathetic Dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) | BERTScore gold references | CC BY-NC 4.0 |

---

## Evaluation Status

| Task | Status |
|---|---|
| Emotion Classifier (RoBERTa F1 = 0.7127) | ✅ Complete |
| Safety Guardrail (DeBERTa recall = 0.9629) | ✅ Complete |
| BERTScore Evaluation (F1 = 0.8266) | ✅ Complete |
| Wilcoxon Test (p = 3.62e-08) | ✅ Complete |
| Adversarial Probe Evaluation (30 probes) | ✅ Complete |
| Human Evaluation (8–10 raters) | 🔄 In Progress |
| Loom Demo Recording | 🔄 In Progress |
| Final Report | 🔄 In Progress |

---

## Known Limitations

- **Bait-and-switch probes:** Positive openers cause the guardrail to misclassify follow-up crisis content (40% recall). Most dangerous documented failure mode.
- **Domain transfer false positives:** Academic hyperbole ("this thesis is killing me") fires the guardrail at high confidence — trained on r/SuicideWatch, never saw graduate student language.
- **Generator quality:** Mistral 7B Q4_K_M produces adequate but not optimal empathetic responses. Architecture supports drop-in model replacement.
- **Faithfulness gap:** No automated faithfulness metric deployed — RAGAS and DeepEval both produced degenerate scores with a small model judge. BERTScore reported instead.

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

Dataset licenses vary — see Datasets table above. Mistral 7B: Apache 2.0.
