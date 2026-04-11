# EmpathRAG

**Emotion-Aware Retrieval-Augmented Generation with Safety Guardrails for Student Mental Health**

University of Maryland - MSML641 · Applied Machine Learning · Spring 2025

---

## Overview

EmpathRAG is a 5-stage NLP pipeline that treats a student's emotional state as a first-class signal for retrieval, routing, and safety. Unlike standard RAG systems that process every message identically, EmpathRAG classifies emotion before retrieval, rewrites queries based on emotional context, and intercepts crisis-level messages before the language model is ever invoked.

**Three novel contributions:**
1. Emotion-conditioned retrieval routing with session trajectory tracking
2. Trainable NLI safety guardrail — empirically compared against keyword filter baseline on 30 adversarial probes
3. Integrated Gradients explainability — every guardrail decision is auditable, not a black box

---

## Pipeline Architecture

```
Student Message
      │
      ├──► Stage 1: Emotion Classifier  (RoBERTa + LoRA, CPU)
      │         5 classes: distress / anxiety / frustration / neutral / hopeful
      │
      ├──► Stage 2: Safety Guardrail    (DeBERTa-v3 NLI, CPU)
      │         Crisis detection with Integrated Gradients token attribution
      │         If crisis → return safe response + 988 lifeline, skip generator
      │
      ├──► Stage 3: Query Router        (deterministic template + session tracker)
      │         Emotion-conditioned query rewriting
      │         Trajectory tracking: stable / escalating / de_escalating / volatile
      │
      ├──► Stage 4: FAISS Retrieval     (all-mpnet-base-v2, 768d, GPU→CPU offload)
      │         1,674,369 vectors from Reddit Mental Health corpus
      │         Re-rank by emotion match bonus + safety score
      │
      └──► Stage 5: Mistral 7B Generator (Q4_K_M GGUF, 28/33 GPU layers)
                Sliding window conversation memory (last 3 turns)
                Peer support response: validate emotion → open question
```

**VRAM budget (RTX 3060 6 GB):**

| Component | Device | VRAM |
|---|---|---|
| RoBERTa + DeBERTa | CPU | 0 MB |
| Sentence transformer | GPU (encode only) | 440 MB peak |
| Mistral 7B Q4_K_M | GPU resident | 3,870 MB |
| KV cache + compute | GPU | 405 MB |
| **Total peak** | | **~4,715 MB** |

---

## Evaluation Results

| Metric | Value | Target | Status |
|---|---|---|---|
| RoBERTa Weighted F1 | 0.7127 | > 0.55 | PASS |
| DeBERTa Recall (crisis) | 0.9629 | > 0.80 | PASS +16pts |
| DeBERTa Precision (crisis) | 0.7951 | > 0.65 | PASS +14pts |
| BERTScore F1 | 0.8266 | > 0.72 | PASS |
| Wilcoxon p-value (D vs A) | 3.62e-08 | < 0.05 | DECISIVE |
| Adversarial crisis recall (DeBERTa) | 85% | > 80% | PASS |
| Adversarial crisis recall (keyword) | 45% | — | Baseline |
| Euphemistic recall (DeBERTa) | 100% | — | vs 20% keyword |

**Ablation — Emotion Alignment Score:**

| Condition | Mean Alignment | Description |
|---|---|---|
| A — BM25 baseline | 0.30 | Sparse retrieval, no emotion |
| C — Dense RAG | 0.50 | Semantic retrieval, no emotion conditioning |
| D — Full EmpathRAG | 0.88 | Emotion-conditioned retrieval + routing |

Emotion conditioning contributes +38 points over pure dense retrieval (C→D). Wilcoxon signed-rank test: p = 3.62e-08.

---

## Repository Structure

```
Empath-RAG/
├── src/
│   ├── pipeline/
│   │   ├── pipeline.py          # 5-stage orchestrator with conversation memory
│   │   ├── query_router.py      # Emotion-conditioned query rewriting
│   │   └── session_tracker.py   # Trajectory detection
│   ├── models/
│   │   ├── guardrail_ig.py      # DeBERTa NLI + Integrated Gradients
│   │   └── annotate_corpus.py   # Corpus emotion annotation
│   └── data/
│       ├── preprocess.py        # GoEmotions 27→5 label mapping, text cleaning
│       ├── build_faiss_index.py # FAISS index construction from Reddit corpus
│       ├── build_nli_pairs.py   # NLI training pair construction
│       └── download_datasets.py
├── demo/
│   └── app.py                   # Gradio demo: emotion timeline + crisis panel
├── eval/
│   ├── test_prompts.json        # 50 prompts (10 per emotion class)
│   ├── adversarial_probes.json  # 30 probes across 6 categories
│   ├── run_ablation.py          # Ablation study (Conditions A / C / D)
│   ├── run_bertscore.py         # BERTScore F1 vs Empathetic Dialogues gold
│   ├── run_wilcoxon.py          # Wilcoxon signed-rank test
│   └── run_adversarial.py       # DeBERTa vs keyword filter comparison
├── notebooks/
│   ├── colab_emotion_classifier.ipynb   # RoBERTa + LoRA training (A100)
│   └── colab_deberta_guardrail.ipynb    # DeBERTa NLI training (A100)
├── requirements.txt             # Pinned dependencies with install order
└── data/MANIFEST.md             # Dataset download instructions
```

Models and data are not included in the repository. See [Setup](#setup) below.

---

## Setup

**Requirements:** Python 3.12, CUDA 12.1, 6 GB VRAM, 16 GB RAM, ~15 GB disk

**Install order is critical — do not skip steps:**

```bash
# 1. Clone
git clone https://github.com/MukulRay1603/Empath-RAG.git
cd Empath-RAG
python -m venv venv
venv\Scripts\activate   # Windows

# 2. PyTorch with CUDA — install FIRST
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. llama-cpp-python CUDA wheel — install SECOND
pip install "llama_cpp_python==0.3.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 4. All other dependencies
pip install -r requirements.txt

# 5. Force correct numpy (faiss requires <2.0)
pip install "numpy==1.26.4" --force-reinstall
```

**Download models and data** — see `data/MANIFEST.md` for dataset download instructions.

Mistral 7B GGUF: download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` from [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) and place at `models/generator/mistral-7b-instruct-v0.2.Q4_K_M.gguf`.

---

## Running the Demo

```bash
python demo/app.py
```

Opens at `http://127.0.0.1:7860` with a public Gradio tunnel URL. Features:
- Real-time emotion timeline with trajectory detection
- Safety guardrail panel with Integrated Gradients token highlights
- Sliding 3-turn conversation memory
- Session ID for human evaluation correlation
- Automatic session logging to `eval/human_eval_log.jsonl`

---

## Datasets

| Dataset | Role | License | Size |
|---|---|---|---|
| GoEmotions (Google) | Emotion classifier training | Apache 2.0 | 43,410 Reddit comments, 27 labels |
| Reddit Mental Health (Zenodo 3941387) | FAISS retrieval corpus | CC BY 4.0 | 108 CSVs, 3.1 GB |
| Suicide Detection (Kaggle) | Guardrail NLI training | Public | 232,074 posts |
| Empathetic Dialogues (Meta) | BERTScore references | CC BY-NC 4.0 | 76,673 turns |

All datasets are publicly licensed. See `data/MANIFEST.md` for download commands.

---

## Known Limitations

- **Generator quality:** Mistral 7B Q4_K_M (4-bit quantized) produces adequate but not optimal empathetic responses. A larger or mental-health fine-tuned model would substantially improve quality.
- **Session memory only:** Conversation history is maintained within a session but not persisted across sessions.
- **Domain transfer:** DeBERTa guardrail trained on r/SuicideWatch produces false positives on academic stress hyperbole ("this thesis is killing me"). Documented in adversarial evaluation.

---

## License

MIT License. See `LICENSE` for details.

Dataset licenses vary — see Datasets table above. Mistral 7B: Apache 2.0.
