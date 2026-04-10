"""
eval/run_ablation.py
Ablation study: compare Condition A (BM25), C (Dense RAG no emotion), D (Full EmpathRAG).
Computes Condition C as TRUE no-emotion-conditioning ablation:
  - No emotion query rewriting (raw user_message goes to FAISS)
  - No re-ranking at all - pure FAISS distance order, no safety score, no emotion signal
Loads Conditions A and D from eval/wilcoxon_results.json.
"""

import sys, json, types
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "eval")

import numpy as np
import sqlite3
import torch
import time
from pipeline.pipeline import EmpathRAGPipeline, SAFE_RESPONSE, LABEL_NAMES
from pipeline.query_router import route_query

PROMPTS_PATH = "eval/test_prompts.json"
WILCOXON_PATH = "eval/wilcoxon_results.json"
RESULTS_PATH = "eval/ablation_results.json"


def add_condition_c_methods(pipeline):
    """
    Adds two methods to pipeline instance for Condition C ablation:
    1. _retrieve_no_emotion: pure FAISS distance order, no re-ranking, no emotion or safety score
    2. run_condition_c: full pipeline run with raw user_message and no emotion conditioning
    """

    def _retrieve_no_emotion(self, query: str, emotion_label: int) -> list[str]:
        """
        Pure semantic retrieval - no emotion conditioning of any kind.
        Returns top_k chunks in FAISS distance order (closest first).
        No re-ranking, no safety score, no emotion bonus.
        emotion_label parameter accepted but deliberately ignored.
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

        # Search top_k directly - no need for top_k*3 since we are not re-ranking
        distances, ids = self.faiss_index.search(
            q_vec.astype(np.float32), self.top_k
        )
        # ids[0] is already sorted by L2 distance ascending (closest first)
        # Filter out -1 padding (FAISS uses -1 for unfilled slots)
        faiss_ordered_ids = [int(i) for i in ids[0] if i >= 0]

        if not faiss_ordered_ids:
            return []

        # Fetch text from SQLite - NOTE: SQLite WHERE IN does NOT preserve input order
        placeholders = ",".join("?" * len(faiss_ordered_ids))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            faiss_ordered_ids,
        ).fetchall()
        conn.close()

        # Restore FAISS distance order using id->text map
        id_to_text = {r[0]: r[1] for r in rows}
        # Return in FAISS order, skip any ids not found in SQLite
        return [id_to_text[i] for i in faiss_ordered_ids if i in id_to_text]

    def run_condition_c(self, user_message: str) -> dict:
        """
        Condition C: No emotion-conditioned retrieval.
        Exact copy of real run() with two changes:
        1. guardrail.check has skip_ig=True
        2. Stage 4 uses _retrieve_no_emotion(user_message) instead of _retrieve(routed_query)
        """
        latency = {}
        token_count = len(user_message.split())
        t0 = time.perf_counter()
        emotion_label = self._classify_emotion(user_message)
        latency["emotion_ms"] = round((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        is_crisis, confidence, ig_highlights = self.guardrail.check(
            user_message, threshold=self.guardrail_threshold, skip_ig=True
        )
        latency["guardrail_ms"] = round((time.perf_counter() - t0) * 1000)
        self.tracker.update(emotion_label, token_count)
        trajectory = self.tracker.trajectory()
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
        t0 = time.perf_counter()
        routed_query = route_query(user_message, emotion_label, trajectory)
        latency["router_ms"] = round((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        chunks = self._retrieve_no_emotion(user_message, emotion_label)
        latency["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        response = self._generate(user_message, chunks)
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

    # Bind methods to pipeline instance
    pipeline._retrieve_no_emotion = types.MethodType(_retrieve_no_emotion, pipeline)
    pipeline.run_condition_c = types.MethodType(run_condition_c, pipeline)


def compute_alignment_scores(pipeline, prompts, use_condition_c=False):
    """
    For each non-crisis prompt, compute binary alignment score:
    1 if emotion(query) == emotion(top retrieved chunk), else 0.
    """
    scores = []
    for i, prompt in enumerate(prompts, 1):
        if use_condition_c:
            result = pipeline.run_condition_c(prompt["text"])
        else:
            result = pipeline.run(prompt["text"])

        if result["crisis"]:
            print(f"  Prompt {i:02d}/50: CRISIS (guardrail fired unexpectedly), alignment=0")
            scores.append(0)
            continue

        if not result["retrieved_chunks"]:
            print(f"  WARNING: Prompt {i:02d}/50: NO CHUNKS retrieved, alignment=0")
            scores.append(0)
            continue

        q_emotion = result["emotion"]
        top_chunk = result["retrieved_chunks"][0]
        chunk_emotion = pipeline._classify_emotion(top_chunk)
        alignment = int(q_emotion == chunk_emotion)
        scores.append(alignment)
        print(f"  Prompt {i:02d}/50: alignment={alignment} (query={q_emotion}, chunk={chunk_emotion})")

    return scores


def run_ablation_eval():
    # Load test prompts
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    # Load Conditions A and D from Wilcoxon results
    print("Loading Conditions A and D from wilcoxon_results.json...")
    with open(WILCOXON_PATH) as f:
        wilcoxon = json.load(f)

    scores_a = wilcoxon["condition_a_scores"]
    scores_d = wilcoxon["condition_d_scores"]
    print(f"  Condition A (BM25): {len(scores_a)} scores loaded")
    print(f"  Condition D (Full EmpathRAG): {len(scores_d)} scores loaded")

    # Compute Condition C: Dense RAG without emotion conditioning
    print("\nCondition C - Dense RAG without emotion conditioning")
    print("Initializing pipeline (use_real_guardrail=False)...")
    pipeline = EmpathRAGPipeline(use_real_guardrail=False, guardrail_threshold=0.5)

    # Add Condition C methods
    print("Adding Condition C methods (no query rewriting, no emotion bonus)...")
    add_condition_c_methods(pipeline)

    print("Computing Condition C alignment scores...")
    scores_c = compute_alignment_scores(pipeline, prompts, use_condition_c=True)

    # Compute means
    mean_a = sum(scores_a) / len(scores_a)
    mean_c = sum(scores_c) / len(scores_c)
    mean_d = sum(scores_d) / len(scores_d)

    # Print summary table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Condition':<30} | {'Mean Alignment':>15} | {'N':>3}")
    print("-"*60)
    print(f"{'A (BM25 baseline)':<30} | {mean_a:>15.4f} | {len(scores_a):>3}")
    print(f"{'C (Dense RAG, no emotion)':<30} | {mean_c:>15.4f} | {len(scores_c):>3}")
    print(f"{'D (Full EmpathRAG)':<30} | {mean_d:>15.4f} | {len(scores_d):>3}")
    print("="*60)

    # Save results
    output = {
        "condition_a_scores": scores_a,
        "condition_c_scores": scores_c,
        "condition_d_scores": scores_d,
        "condition_a_mean": round(mean_a, 4),
        "condition_c_mean": round(mean_c, 4),
        "condition_d_mean": round(mean_d, 4),
        "n": len(prompts),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_ablation_eval()
