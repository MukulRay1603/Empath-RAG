"""
src/pipeline/pipeline.py
EmpathRAG — Full 5-Stage Pipeline Orchestrator

Stage 1: Emotion Classifier  (RoBERTa + LoRA, CPU)
Stage 2: Safety Guardrail    (DeBERTa NLI stub until checkpoint lands)
Stage 3: Query Router        (deterministic template, uses session trajectory)
Stage 4: FAISS Retrieval     (all-mpnet-base-v2, GPU→CPU offload)
Stage 5: Mistral 7B Generator (GGUF, 28 GPU layers)

VRAM sequencing on RTX 3060 6GB:
  - RoBERTa + DeBERTa: CPU only — 0 MB GPU
  - Sentence transformer: load to GPU (~440 MB), run query, offload back to CPU
  - Mistral: always resident on GPU (~3,870 MB)
  - Peak during retrieval: ~440 MB (ST) — Mistral not yet loaded at that point
  - Peak during generation: ~3,870 MB (Mistral) — ST already offloaded
  - Absolute peak: ~4,310 MB — well within 6 GB ceiling
"""

import asyncio
import sqlite3
import time
import torch
import numpy as np
import faiss

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from .session_tracker import SessionTracker
from .query_router import route_query, LABEL_NAMES


# ── Constants ─────────────────────────────────────────────────────────────────

SAFE_RESPONSE = (
    "I can hear that you're going through something really difficult. "
    "You're not alone in this. Please reach out to the 988 Suicide "
    "& Crisis Lifeline by calling or texting 988 — they're available "
    "24/7. If you're a student, your campus counseling center is "
    "also a great resource. Would you like help finding their contact info?"
)

SYSTEM_PROMPT = """You are a warm peer companion for graduate students. You write like a caring friend in a chat window.

Here is an example of a perfect response:
Student: "I've been feeling really anxious about my qualifying exam."
Response: "Qualifying exams are genuinely terrifying, and that knot-in-your-stomach feeling is something so many grad students know too well.\n\nWhat's been weighing on you most about it?"

Follow that exact format every time. Two short paragraphs separated by a blank line. First paragraph: one warm sentence that reflects their exact feeling back to them using their own words - make it feel human, not clinical. Second paragraph: one gentle open question starting with What or What's that invites them to share more. Nothing else. No advice. No suggestions. No lists. No context references."""

DEBERTA_HYPOTHESIS = "This person is expressing suicidal ideation or intent to self-harm."

# Safety score by emotion label — used during emotion-filtered retrieval scoring
SAFETY_SCORE_MAP = {0: 0.0, 1: 0.0, 2: 0.3, 3: 0.7, 4: 1.0}


# ── Guardrail stub ─────────────────────────────────────────────────────────────
# Replace this class body with the real DeBERTa implementation once
# Karthik's checkpoint lands in models/safety_guardrail/.
# The stub always returns (False, 0.0, []) — pipeline proceeds normally.
# The interface is identical to the real SafetyGuardrail in guardrail_ig.py
# so the swap is a one-line import change.

class _GuardrailStub:
    """
    Passthrough stub for SafetyGuardrail.
    Returns (is_crisis=False, confidence=0.0, token_attributions=[]).
    Replace with real guardrail once DeBERTa checkpoint is available:

        from src.models.guardrail_ig import SafetyGuardrail
        self.guardrail = SafetyGuardrail()
    """
    def check(self, text: str, threshold: float = 0.5, skip_ig: bool = False):
        return False, 0.0, []


# ── Pipeline ──────────────────────────────────────────────────────────────────

class EmpathRAGPipeline:
    """
    Full EmpathRAG pipeline.

    Usage:
        pipeline = EmpathRAGPipeline()
        result   = pipeline.run("I feel completely overwhelmed")

    Result dict keys:
        response          str   — generated or safe-template response
        emotion           int   — 0-4 emotion label for this turn
        emotion_name      str   — human-readable label
        trajectory        str   — session trajectory type
        crisis            bool  — True if guardrail fired
        crisis_confidence float — guardrail entailment probability
        ig_highlights     list  — [(token, score), ...] top-5 if crisis
        retrieved_chunks  list  — list of retrieved text strings (empty if crisis)
        latency_ms        dict  — per-stage latency breakdown
    """

    def __init__(
        self,
        ec_checkpoint:   str = "models/emotion_classifier",
        guardrail_ckpt:  str = "models/safety_guardrail",
        faiss_index_path:str = "data/indexes/faiss_flat.index",
        db_path:         str = "data/indexes/metadata.db",
        mistral_path:    str = "models/generator/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        st_model:        str = "sentence-transformers/all-mpnet-base-v2",
        n_gpu_layers:    int = 28,
        n_ctx:           int = 4096,
        top_k:           int = 5,
        tracker_n:       int = 3,
        guardrail_threshold: float = 0.5,
        use_real_guardrail:  bool  = False,
    ):
        self.top_k               = top_k
        self.guardrail_threshold = guardrail_threshold
        self.db_path             = db_path

        print("[EmpathRAG] Loading emotion classifier (CPU)...")
        self.ec_tok   = AutoTokenizer.from_pretrained(ec_checkpoint)
        _base         = AutoModelForSequenceClassification.from_pretrained(
                            "roberta-base", num_labels=5
                        )
        self.ec_model = PeftModel.from_pretrained(_base, ec_checkpoint).eval()
        # Explicitly keep on CPU — no .to("cuda")
        print(f"[EmpathRAG] Emotion classifier ready. "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        print("[EmpathRAG] Loading safety guardrail...")
        if use_real_guardrail:
            # Swap in real guardrail once DeBERTa checkpoint exists
            try:
                try:
                    from src.models.guardrail_ig import SafetyGuardrail
                except ImportError:
                    from models.guardrail_ig import SafetyGuardrail
                self.guardrail = SafetyGuardrail()
                print("[EmpathRAG] Real DeBERTa guardrail loaded (CPU).")
            except Exception as e:
                print(f"[EmpathRAG] WARNING: Real guardrail failed to load ({e}). "
                      f"Falling back to stub.")
                self.guardrail = _GuardrailStub()
        else:
            self.guardrail = _GuardrailStub()
            print("[EmpathRAG] Guardrail stub active — swap to real once "
                  "models/safety_guardrail/ is populated.")

        print("[EmpathRAG] Loading sentence transformer (will GPU-offload after each query)...")
        self.st_model_name = st_model
        self.encoder       = SentenceTransformer(st_model, device="cpu")
        # Start on CPU — we move to GPU only during encode(), then back

        print("[EmpathRAG] Loading FAISS index...")
        self.faiss_index = faiss.read_index(faiss_index_path)
        print(f"[EmpathRAG] FAISS: {self.faiss_index.ntotal:,} vectors")

        print("[EmpathRAG] Loading Mistral 7B (GPU)...")
        self.llm = Llama(
            model_path   = mistral_path,
            n_ctx        = n_ctx,
            n_gpu_layers = n_gpu_layers,
            verbose      = False,
        )
        print(f"[EmpathRAG] Mistral ready. "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        self.tracker = SessionTracker(N=tracker_n)
        self.conv_history = []  # list of {"role": "user"|"assistant", "content": str}
        print("[EmpathRAG] Pipeline initialised. Ready for inference.")

    # ── Stage 1: Emotion classification ───────────────────────────────────────

    def _classify_emotion(self, text: str) -> int:
        """Returns integer emotion label 0-4. Runs on CPU."""
        enc = self.ec_tok(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.ec_model(**enc).logits
        return int(logits.argmax(-1).item())

    # ── Stage 4: FAISS retrieval ───────────────────────────────────────────────

    def _retrieve(self, query: str, emotion_label: int) -> list[str]:
        """
        Encodes query on GPU, searches FAISS, filters via SQLite.
        Returns top_k chunk texts ranked by emotion match + safety score.
        GPU usage: ~440 MB during encode, freed before returning.
        """
        # Move encoder to GPU for this call only
        self.encoder.to("cuda")
        q_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Immediately offload back to CPU
        self.encoder.to("cpu")
        torch.cuda.empty_cache()

        # Search wider than top_k so we have room to re-rank by emotion
        distances, ids = self.faiss_index.search(
            q_vec.astype(np.float32), self.top_k * 3
        )
        candidate_ids = [int(i) for i in ids[0] if i >= 0]

        if not candidate_ids:
            return []

        # Fetch metadata from SQLite
        placeholders = ",".join("?" * len(candidate_ids))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"SELECT id, text, emotion_label, safety_score FROM chunks "
            f"WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        conn.close()

        # Re-rank: emotion match gets +2.0 bonus, then by safety_score
        def _score(row):
            _, _, chunk_emotion, safety = row
            match_bonus = 2.0 if chunk_emotion == emotion_label else 0.0
            return match_bonus + safety

        rows_sorted = sorted(rows, key=_score, reverse=True)[: self.top_k]
        return [r[1] for r in rows_sorted]

    # ── Stage 5: Generation ────────────────────────────────────────────────────

    def _generate(self, user_message: str, chunks: list[str]) -> str:
        """Generates empathetic response conditioned on retrieved context and conversation history."""
        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

        # Build Mistral multi-turn prompt with conversation history
        # Mistral Instruct format: <s>[INST] msg [/INST] response</s>[INST] msg [/INST]
        prompt_parts = []

        # First turn always includes system prompt and context
        first_user = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context (for emotional grounding only - never reference this directly):\n{context}\n\n"
            f"Student: {user_message}"
        )

        if not self.conv_history:
            # No history - simple single turn
            prompt = f"[INST] {first_user}\n\nResponse: [/INST]"
        else:
            # Build multi-turn prompt from history
            # History entries alternate user/assistant
            prompt = "<s>"
            for i, entry in enumerate(self.conv_history):
                if entry["role"] == "user":
                    # Include system prompt only on first historical turn
                    if i == 0:
                        turn_content = (
                            f"{SYSTEM_PROMPT}\n\n"
                            f"Context (for emotional grounding only - never reference this directly):\n{context}\n\n"
                            f"Student: {entry['content']}"
                        )
                    else:
                        turn_content = f"Student: {entry['content']}"
                    prompt += f"[INST] {turn_content} [/INST]"
                else:
                    # assistant turn
                    prompt += f" {entry['content']}</s>"
            # Append current user message
            prompt += f"[INST] Student: {user_message}\n\nResponse: [/INST]"

        out = self.llm(
            prompt,
            max_tokens  = 200,
            temperature = 0.75,
            stop        = ["[INST]", "Student:", "\n\n\n", "</s>"],
        )
        raw = out["choices"][0]["text"].strip()

        # Ensure blank line between the two paragraphs for Gradio rendering
        if "\n\n" in raw:
            return raw
        # Otherwise find the question sentence and split into two paragraphs
        if "?" in raw:
            sentences = raw.replace("  ", " ").split(". ")
            question_idx = None
            for i, s in enumerate(sentences):
                if "?" in s:
                    question_idx = i
                    break
            if question_idx is not None and question_idx > 0:
                para1 = ". ".join(sentences[:question_idx]).strip()
                if not para1.endswith("."):
                    para1 += "."
                para2 = sentences[question_idx].strip()
                if not para2.endswith("?"):
                    para2 += "?"
                return para1 + "\n\n" + para2
        return raw

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, user_message: str) -> dict:
        """
        Run the full 5-stage pipeline on a single user message.

        Returns structured dict — see class docstring for keys.
        """
        latency = {}
        token_count = len(user_message.split())

        # ── Stage 1: Emotion classification ───────────────────────────────────
        t0 = time.perf_counter()
        emotion_label = self._classify_emotion(user_message)
        latency["emotion_ms"] = round((time.perf_counter() - t0) * 1000)

        # ── Stage 2: Safety guardrail ──────────────────────────────────────────
        t0 = time.perf_counter()
        is_crisis, confidence, ig_highlights = self.guardrail.check(
            user_message, threshold=self.guardrail_threshold
        )
        latency["guardrail_ms"] = round((time.perf_counter() - t0) * 1000)

        # Update session tracker (skip very short filler messages)
        self.tracker.update(emotion_label, token_count)
        trajectory = self.tracker.trajectory()

        # ── Guardrail intercept: terminate pipeline, return safe response ──────
        if is_crisis:
            return {
                "response":          SAFE_RESPONSE,
                "emotion":           emotion_label,
                "emotion_name":      LABEL_NAMES[emotion_label],
                "trajectory":        trajectory,
                "crisis":            True,
                "crisis_confidence": confidence,
                "ig_highlights":     ig_highlights,
                "retrieved_chunks":  [],
                "latency_ms":        latency,
            }

        # ── Stage 3: Query routing ─────────────────────────────────────────────
        t0 = time.perf_counter()
        routed_query = route_query(user_message, emotion_label, trajectory)
        latency["router_ms"] = round((time.perf_counter() - t0) * 1000)

        # ── Stage 4: Retrieval ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = self._retrieve(routed_query, emotion_label)
        latency["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)

        # ── Stage 5: Generation ────────────────────────────────────────────────
        t0 = time.perf_counter()
        response = self._generate(user_message, chunks)
        # Update conversation history (keep last 6 entries = 3 turns)
        self.conv_history.append({"role": "user",      "content": user_message})
        self.conv_history.append({"role": "assistant", "content": response})
        if len(self.conv_history) > 6:
            self.conv_history = self.conv_history[-6:]
        latency["generation_ms"] = round((time.perf_counter() - t0) * 1000)

        latency["total_ms"] = sum(latency.values())

        return {
            "response":          response,
            "emotion":           emotion_label,
            "emotion_name":      LABEL_NAMES[emotion_label],
            "trajectory":        trajectory,
            "crisis":            False,
            "crisis_confidence": 0.0,
            "ig_highlights":     [],
            "retrieved_chunks":  chunks,
            "latency_ms":        latency,
        }

    def reset_session(self):
        """Clear session emotion history and conversation history."""
        self.tracker.reset()
        self.conv_history = []
