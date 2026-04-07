"""
eval/build_bertscore_references.py

Pre-compute BERTScore reference responses for all 50 test prompts.
For each prompt, finds the most semantically similar response in the
Empathetic Dialogues test split using cosine similarity.
Saves result to eval/bertscore_references.json.

Run once from repo root:
    python eval/build_bertscore_references.py

Takes ~2 minutes on CPU. Does not require GPU.
"""

import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROMPTS_PATH    = "eval/test_prompts.json"
OUTPUT_PATH     = "eval/bertscore_references.json"
ST_MODEL        = "sentence-transformers/all-mpnet-base-v2"
ED_SPLIT        = "test"   # use held-out split only — no data leakage


def main():
    print("Loading test prompts...")
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"  {len(prompts)} prompts loaded.")

    print("Loading Empathetic Dialogues test split...")
    ed = load_dataset("facebook/empathetic_dialogues", split=ED_SPLIT)
    # ED stores multi-turn conversations. Extract the utterances field
    # which contains the actual text of each turn.
    # We want the *response* turns (index 1, 3, 5...) — the empathetic replies.
    responses = []
    for row in ed:
        utterances = row.get("utterances", "")
        if not utterances:
            # Fall back to 'utterance' field (single turn datasets)
            utt = row.get("utterance", "").strip()
            if utt:
                responses.append(utt)
        else:
            # Multi-turn: take every second utterance (the listener/responder)
            turns = [u.strip() for u in utterances.split("_conv_")
                     if u.strip() and len(u.strip()) > 20]
            responses.extend(turns[1::2])  # odd indices = responses

    # Fallback: if parsing produced nothing, just use all utterances
    if not responses:
        print("  Fallback: using all ED utterances as response pool.")
        for row in ed:
            utt = row.get("utterance", "").strip()
            if utt and len(utt) > 20:
                responses.append(utt)

    # Deduplicate and cap at 20K for speed
    responses = list(dict.fromkeys(responses))[:20_000]
    print(f"  {len(responses):,} candidate responses.")

    print(f"Loading sentence transformer: {ST_MODEL}")
    model = SentenceTransformer(ST_MODEL, device="cpu")

    print("Encoding ED responses (this is the slow part ~90s)...")
    response_vecs = model.encode(
        responses,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Encoding test prompts...")
    prompt_texts = [p["text"] for p in prompts]
    prompt_vecs  = model.encode(
        prompt_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Computing cosine similarities and finding best matches...")
    # Cosine similarity = dot product (vectors are already normalised)
    sim_matrix = prompt_vecs @ response_vecs.T   # (50, N)

    results = []
    for i, prompt in enumerate(prompts):
        best_idx   = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        best_ref   = responses[best_idx]

        results.append({
            "id":            prompt["id"],
            "emotion":       prompt["emotion"],
            "prompt":        prompt["text"],
            "reference":     best_ref,
            "sim_score":     round(best_score, 4),
        })

    # Sanity check — print a few matches
    print("\nSample matches:")
    for r in results[:3]:
        print(f"\n  Prompt    : {r['prompt'][:80]}...")
        print(f"  Reference : {r['reference'][:80]}...")
        print(f"  Sim score : {r['sim_score']:.4f}")

    avg_sim = np.mean([r["sim_score"] for r in results])
    print(f"\nAverage similarity: {avg_sim:.4f}")
    if avg_sim < 0.3:
        print("  ⚠️  Low average similarity — ED may not be a great match for "
              "these prompts, but BERTScore will still work.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved {len(results)} references to {OUTPUT_PATH}")
    print("Ready for BERTScore evaluation.")


if __name__ == "__main__":
    main()
