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
from pathlib import Path
import torch
import numpy as np
import faiss

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from .session_tracker import SessionTracker
from .query_router import route_query, LABEL_NAMES
from .safety_policy import SafetyLevel, SafetyTriagePolicy


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
        retrieval_corpus: str = "reddit_research",
        curated_index_path: str = "data/curated/indexes/faiss_curated.index",
        curated_db_path: str = "data/curated/indexes/metadata_curated.db",
        mistral_path:    str = "models/generator/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        st_model:        str = "sentence-transformers/all-mpnet-base-v2",
        n_gpu_layers:    int = 28,
        n_ctx:           int = 4096,
        generation_max_tokens: int = 200,
        top_k:           int = 5,
        tracker_n:       int = 3,
        guardrail_threshold: float = 0.5,
        use_real_guardrail:  bool  = True,
        allow_stub_guardrail: bool = False,
    ):
        self.top_k               = top_k
        self.generation_max_tokens = generation_max_tokens
        self.guardrail_threshold = guardrail_threshold
        self.retrieval_corpus    = self._resolve_retrieval_corpus(
            retrieval_corpus, curated_index_path, curated_db_path
        )
        self.faiss_index_path    = curated_index_path if self.retrieval_corpus == "curated_support" else faiss_index_path
        self.db_path             = curated_db_path if self.retrieval_corpus == "curated_support" else db_path
        self.safety_policy       = SafetyTriagePolicy(
            support_threshold=guardrail_threshold
        )

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
                if not allow_stub_guardrail:
                    raise RuntimeError(
                        "Real safety guardrail failed to load. EmpathRAG v2 fails "
                        "closed by default; pass allow_stub_guardrail=True only for "
                        "offline development or retrieval-only experiments."
                    ) from e
                print(f"[EmpathRAG] WARNING: Real guardrail failed to load ({e}). "
                      f"Falling back to stub because allow_stub_guardrail=True.")
                self.guardrail = _GuardrailStub()
        else:
            if not allow_stub_guardrail:
                raise RuntimeError(
                    "use_real_guardrail=False disables the crisis guardrail. Pass "
                    "allow_stub_guardrail=True only for controlled development or "
                    "component-level evaluation."
                )
            self.guardrail = _GuardrailStub()
            print("[EmpathRAG] Guardrail stub active — swap to real once "
                  "models/safety_guardrail/ is populated.")

        print("[EmpathRAG] Loading sentence transformer (will GPU-offload after each query)...")
        self.st_model_name = st_model
        self.encoder       = SentenceTransformer(st_model, device="cpu")
        # Start on CPU — we move to GPU only during encode(), then back

        print("[EmpathRAG] Loading FAISS index...")
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        print(f"[EmpathRAG] Retrieval corpus: {self.retrieval_corpus}")
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

    def _resolve_retrieval_corpus(
        self,
        retrieval_corpus: str,
        curated_index_path: str,
        curated_db_path: str,
    ) -> str:
        allowed = {"reddit_research", "curated_support", "auto"}
        if retrieval_corpus not in allowed:
            raise ValueError(f"retrieval_corpus must be one of {sorted(allowed)}")
        if retrieval_corpus == "auto":
            curated_ready = Path(curated_index_path).exists() and Path(curated_db_path).exists()
            return "curated_support" if curated_ready else "reddit_research"
        return retrieval_corpus

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

    def _retrieve(
        self,
        query: str,
        emotion_label: int,
        safety_level: SafetyLevel = SafetyLevel.PASS,
    ) -> list[dict]:
        """
        Encodes query on GPU, searches FAISS, filters via SQLite.
        Returns top_k chunk metadata dicts.
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

        # Search wider than top_k so filters have room to work.
        search_multiplier = 8 if self.retrieval_corpus == "curated_support" else 3
        distances, ids = self.faiss_index.search(
            q_vec.astype(np.float32), self.top_k * search_multiplier
        )
        candidate_ids = [int(i) for i in ids[0] if i >= 0]

        if not candidate_ids:
            return []

        if self.retrieval_corpus == "curated_support":
            return self._fetch_curated_rows(candidate_ids, safety_level=safety_level)

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
        return [
            {
                "id": r[0],
                "text": r[1],
                "emotion_label": r[2],
                "safety_score": r[3],
                "source_name": "Reddit Mental Health",
                "source_type": "research_corpus",
                "title": "Reddit Mental Health chunk",
                "url": "",
                "topic": "",
                "risk_level": "research_only",
                "usage_mode": "retrieval",
            }
            for r in rows_sorted
        ]

    def _fetch_curated_rows(
        self,
        candidate_ids: list[int],
        safety_level: SafetyLevel = SafetyLevel.PASS,
    ) -> list[dict]:
        placeholders = ",".join("?" * len(candidate_ids))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"""
            SELECT id, resource_id, text, source_id, source_name, source_type,
                   title, url, topic, audience, risk_level, usage_mode, summary,
                   last_checked, notes
            FROM chunks
            WHERE id IN ({placeholders})
            """,
            candidate_ids,
        ).fetchall()
        conn.close()

        by_id = {row[0]: row for row in rows}
        ordered = [by_id[i] for i in candidate_ids if i in by_id]
        allowed_usage_modes = self._allowed_curated_usage_modes(safety_level)
        filtered_candidates = [
            row for row in ordered
            if row[10] != "exclude" and row[11] != "metadata_only"
            and row[11] in allowed_usage_modes
        ]
        filtered = self._limit_curated_source_repetition(filtered_candidates)
        return [
            {
                "id": row[0],
                "resource_id": row[1],
                "text": row[2],
                "source_id": row[3],
                "source_name": row[4],
                "source_type": row[5],
                "title": row[6],
                "url": row[7],
                "topic": row[8],
                "audience": row[9],
                "risk_level": row[10],
                "usage_mode": row[11],
                "summary": row[12],
                "last_checked": row[13],
                "notes": row[14],
            }
            for row in filtered
        ]

    def _allowed_curated_usage_modes(self, safety_level: SafetyLevel) -> set[str]:
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return {"crisis_only"}
        if safety_level == SafetyLevel.WELLBEING_SUPPORT:
            return {"retrieval", "wellbeing_only"}
        return {"retrieval"}

    def _limit_curated_source_repetition(self, rows: list[tuple]) -> list[tuple]:
        selected = []
        source_counts: dict[str, int] = {}
        for row in rows:
            source_name = row[4]
            if source_counts.get(source_name, 0) >= 2:
                continue
            selected.append(row)
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            if len(selected) == self.top_k:
                return selected

        if len(selected) < self.top_k:
            selected_ids = {row[0] for row in selected}
            for row in rows:
                if row[0] in selected_ids:
                    continue
                selected.append(row)
                if len(selected) == self.top_k:
                    break
        return selected

    def _retrieve_crisis_support_sources(self, emotion_label: int) -> list[dict]:
        if self.retrieval_corpus != "curated_support":
            return []
        query = (
            "immediate crisis help for a UMD student, 988 Suicide and Crisis "
            "Lifeline, emergency services, after-hours counseling support"
        )
        try:
            return self._retrieve(
                query,
                emotion_label,
                safety_level=SafetyLevel.CRISIS,
            )
        except Exception as exc:
            print(f"[EmpathRAG] WARNING: crisis source retrieval failed: {exc}")
            return []

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
            max_tokens  = self.generation_max_tokens,
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
        safety_decision = self.safety_policy.classify(
            user_message, confidence=confidence, model_flag=is_crisis
        )

        # Update session tracker (skip very short filler messages)
        self.tracker.update(emotion_label, token_count)
        trajectory = self.tracker.trajectory()

        # ── Guardrail intercept: terminate pipeline, return safe response ──────
        if safety_decision.should_intercept:
            t0 = time.perf_counter()
            crisis_sources = self._retrieve_crisis_support_sources(emotion_label)
            latency["crisis_retrieval_ms"] = round((time.perf_counter() - t0) * 1000)
            return {
                "response":          safety_decision.response or SAFE_RESPONSE,
                "emotion":           emotion_label,
                "emotion_name":      LABEL_NAMES[emotion_label],
                "trajectory":        trajectory,
                "crisis":            safety_decision.level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY},
                "crisis_confidence": confidence,
                "safety_level":      safety_decision.level.value,
                "safety_reason":     safety_decision.reason,
                "ig_highlights":     ig_highlights,
                "retrieved_chunks":  [],
                "retrieved_sources": self._source_summaries(crisis_sources),
                "retrieval_corpus":   self.retrieval_corpus,
                "latency_ms":        latency,
            }

        # ── Stage 3: Query routing ─────────────────────────────────────────────
        t0 = time.perf_counter()
        routed_query = route_query(user_message, emotion_label, trajectory)
        latency["router_ms"] = round((time.perf_counter() - t0) * 1000)

        # ── Stage 4: Retrieval ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        retrieved = self._retrieve(
            routed_query,
            emotion_label,
            safety_level=safety_decision.level,
        )
        chunks = [row["text"] for row in retrieved]
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
            "safety_level":      safety_decision.level.value,
            "safety_reason":     safety_decision.reason,
            "ig_highlights":     [],
            "retrieved_chunks":  chunks,
            "retrieved_sources": self._source_summaries(retrieved),
            "retrieval_corpus":  self.retrieval_corpus,
            "latency_ms":        latency,
        }

    def _source_summaries(self, retrieved: list[dict]) -> list[dict]:
        return [
            {
                "title": row.get("title", ""),
                "source_name": row.get("source_name", ""),
                "url": row.get("url", ""),
                "topic": row.get("topic", ""),
                "risk_level": row.get("risk_level", ""),
                "usage_mode": row.get("usage_mode", ""),
            }
            for row in retrieved
        ]

    def reset_session(self):
        """Clear session emotion history and conversation history."""
        self.tracker.reset()
        self.conv_history = []
