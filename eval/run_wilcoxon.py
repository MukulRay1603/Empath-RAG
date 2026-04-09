"""
eval/run_wilcoxon.py
Wilcoxon signed-rank test: Condition D (EmpathRAG) vs Condition A (BM25 baseline).
Tests whether emotion-conditioned retrieval produces statistically significantly
higher emotion alignment scores than vanilla BM25 (p < 0.05).

Emotion alignment score: binary 1/0 per prompt — 1 if query emotion label matches
the emotion label of the top retrieved chunk, 0 otherwise.
"""

import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, "eval")

import numpy as np
from scipy.stats import wilcoxon
from pipeline.pipeline import EmpathRAGPipeline

PROMPTS_PATH = "eval/test_prompts.json"
RESULTS_PATH = "eval/wilcoxon_results.json"

def compute_alignment_scores(pipeline, prompts):
    """
    For each non-crisis prompt, compute binary alignment score:
    1 if emotion(query) == emotion(top retrieved chunk), else 0.
    """
    scores = []
    for prompt in prompts:
        result = pipeline.run(prompt["text"])
        if result["crisis"] or not result["retrieved_chunks"]:
            continue
        q_emotion   = result["emotion"]
        top_chunk   = result["retrieved_chunks"][0]
        chunk_emotion = pipeline._classify_emotion(top_chunk)
        scores.append(int(q_emotion == chunk_emotion))
    return scores

def run_wilcoxon_eval():
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    # ── Condition D: full EmpathRAG pipeline ──────────────────────────────────
    print("Condition D — Full EmpathRAG pipeline")
    # use_real_guardrail=False: Wilcoxon tests RETRIEVAL quality (Stage 4),
    # not guardrail behavior (Stage 2). With real guardrail at t=0.50, 37/50
    # prompts are intercepted before retrieval — leaving only 13 samples.
    # Guardrail and retrieval are independent components; disabling guardrail
    # here lets all 50 prompts reach the retrieval stage as intended.
    pipeline_d = EmpathRAGPipeline(use_real_guardrail=False, guardrail_threshold=0.5)
    original_check = pipeline_d.guardrail.check
    def fast_check(text, threshold=0.5, skip_ig=False):
        return original_check(text, threshold=threshold, skip_ig=True)
    pipeline_d.guardrail.check = fast_check

    print("Computing Condition D alignment scores...")
    scores_d = compute_alignment_scores(pipeline_d, prompts)
    print(f"  D alignment: {np.mean(scores_d):.3f} ({sum(scores_d)}/{len(scores_d)} prompts aligned)")

    # ── Condition A: BM25 baseline ────────────────────────────────────────────
    # We reuse pipeline_d for emotion classification and swap out _retrieve
    # to use BM25 instead of FAISS+emotion-filtering
    print("\nCondition A — BM25 baseline retrieval")
    print("Building BM25 index (this takes ~60-90s)...")
    import condition_a
    bm25, bm25_ids, bm25_texts = condition_a.load_bm25_index()
    print("BM25 index ready.")

    # Monkey-patch _retrieve on pipeline_d to use BM25
    original_retrieve = pipeline_d._retrieve
    def bm25_retrieve(query, emotion_label):
        return condition_a.retrieve_bm25(query, bm25, bm25_ids, bm25_texts, top_k=5)
    pipeline_d._retrieve = bm25_retrieve

    print("Computing Condition A alignment scores...")
    pipeline_d.tracker.reset()
    scores_a = compute_alignment_scores(pipeline_d, prompts)
    print(f"  A alignment: {np.mean(scores_a):.3f} ({sum(scores_a)}/{len(scores_a)} prompts aligned)")

    # Restore original retrieve
    pipeline_d._retrieve = original_retrieve

    # ── Wilcoxon test ─────────────────────────────────────────────────────────
    print("\nRunning Wilcoxon signed-rank test (D vs A, alternative=greater)...")
    # Pad to equal length if needed (should be equal since same prompts)
    min_len = min(len(scores_d), len(scores_a))
    s_d = scores_d[:min_len]
    s_a = scores_a[:min_len]

    if sum(s_d) == sum(s_a):
        print("WARNING: scores are identical — Wilcoxon test not applicable.")
        stat, p_val = float("nan"), float("nan")
    else:
        try:
            # zero_method=pratt handles tied differences correctly for binary 0/1 scores
            stat, p_val = wilcoxon(s_d, s_a, alternative="greater", zero_method="pratt")
        except ValueError as e:
            print(f"WARNING: Wilcoxon failed ({e}) — scores may be too similar.")
            stat, p_val = float("nan"), float("nan")

    print(f"\nWilcoxon Results:")
    print(f"  D mean alignment: {np.mean(s_d):.4f}")
    print(f"  A mean alignment: {np.mean(s_a):.4f}")
    print(f"  Statistic: {stat}")
    print(f"  p-value:   {p_val:.4f}")
    if not np.isnan(p_val):
        print(f"  {'SIGNIFICANT (p < 0.05)' if p_val < 0.05 else 'NOT SIGNIFICANT (p >= 0.05)'}")

    output = {
        "condition_d_mean": round(float(np.mean(s_d)), 4),
        "condition_a_mean": round(float(np.mean(s_a)), 4),
        "condition_d_scores": s_d,
        "condition_a_scores": s_a,
        "wilcoxon_statistic": float(stat) if not np.isnan(stat) else None,
        "p_value":            float(p_val) if not np.isnan(p_val) else None,
        "significant":        bool(p_val < 0.05) if not np.isnan(p_val) else None,
        "n":                  min_len,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_wilcoxon_eval()
