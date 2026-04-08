"""
eval/run_bertscore.py
Compute BERTScore F1 between EmpathRAG generated responses and
gold Empathetic Dialogues references.
Uses pre-computed bertscore_references.json (50 ED gold responses).
Saves results to eval/bertscore_results.json
"""

import sys, json
sys.path.insert(0, "src")

from bert_score import score as bertscore
from pipeline.pipeline import EmpathRAGPipeline

REFS_PATH    = "eval/bertscore_references.json"
PROMPTS_PATH = "eval/test_prompts.json"
RESULTS_PATH = "eval/bertscore_results.json"

def run_bertscore_eval():
    with open(REFS_PATH)    as f: refs_data    = json.load(f)
    with open(PROMPTS_PATH) as f: prompts_data = json.load(f)

    # refs_data is a list of {id, emotion, prompt, reference, sim_score}
    # Build lookup: id -> reference
    ref_lookup = {r["id"]: r["reference"] for r in refs_data}

    print("Initialising pipeline...")
    pipeline = EmpathRAGPipeline(use_real_guardrail=True, guardrail_threshold=0.5)

    # Monkey-patch to skip IG (speed)
    original_check = pipeline.guardrail.check
    def fast_check(text, threshold=0.5, skip_ig=False):
        return original_check(text, threshold=threshold, skip_ig=True)
    pipeline.guardrail.check = fast_check

    candidates = []
    references = []
    skipped    = []

    print(f"Running pipeline on {len(prompts_data)} prompts...")
    for i, prompt in enumerate(prompts_data):
        pid  = prompt["id"]
        text = prompt["text"]

        if pid not in ref_lookup:
            skipped.append(pid)
            continue

        result    = pipeline.run(text)
        candidate = result["response"]
        reference = ref_lookup[pid]

        candidates.append(candidate)
        references.append(reference)

        emotion = result["emotion_name"]
        crisis  = result["crisis"]
        print(f"  [{i+1:02d}] {emotion:<12} crisis={crisis} | {text[:50]}...")

    print(f"\nSkipped {len(skipped)} prompts (no reference found)")
    print(f"Computing BERTScore on {len(candidates)} pairs...")

    P, R, F1 = bertscore(candidates, references, lang="en", verbose=False)

    mean_f1 = float(F1.mean())
    mean_p  = float(P.mean())
    mean_r  = float(R.mean())

    print(f"\nBERTScore Results:")
    print(f"  Precision: {mean_p:.4f}")
    print(f"  Recall:    {mean_r:.4f}")
    print(f"  F1:        {mean_f1:.4f}  (target: > 0.72)")
    print(f"  PASS" if mean_f1 >= 0.72 else f"  BELOW TARGET (target 0.72)")

    per_prompt = [
        {"prompt_id": prompts_data[i]["id"], "f1": round(float(F1[i]), 4),
         "precision": round(float(P[i]), 4), "recall": round(float(R[i]), 4)}
        for i in range(len(candidates))
    ]

    output = {
        "mean_precision": round(mean_p, 4),
        "mean_recall":    round(mean_r, 4),
        "mean_f1":        round(mean_f1, 4),
        "target":         0.72,
        "pass":           mean_f1 >= 0.72,
        "n_evaluated":    len(candidates),
        "n_skipped":      len(skipped),
        "per_prompt":     per_prompt,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_bertscore_eval()
