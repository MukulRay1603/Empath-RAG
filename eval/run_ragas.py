"""
eval/run_ragas.py
RAGAS faithfulness evaluation using local Mistral 7B as judge (no OpenAI).
Evaluates whether generated responses are faithful to retrieved context.
Saves results to eval/ragas_results.json

RAGAS 0.1.21 API:
  - ragas.evaluate(Dataset, metrics=[faithfulness])
  - Dataset needs columns: question, answer, contexts (list of strings)
  - LLM judge configured via ragas.llms with LangchainLLMWrapper
"""

import sys, json
sys.path.insert(0, "src")

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from pipeline.pipeline import EmpathRAGPipeline

PROMPTS_PATH = "eval/test_prompts.json"
RESULTS_PATH = "eval/ragas_results.json"
MISTRAL_PATH = "models/generator/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
N_EVAL       = 40   # evaluate first 40 non-crisis prompts

def run_ragas_eval():
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    print("Initialising pipeline...")
    pipeline = EmpathRAGPipeline(use_real_guardrail=True, guardrail_threshold=0.5)

    original_check = pipeline.guardrail.check
    def fast_check(text, threshold=0.5, skip_ig=False):
        return original_check(text, threshold=threshold, skip_ig=True)
    pipeline.guardrail.check = fast_check

    print("Configuring RAGAS to reuse pipeline Mistral instance (avoids double VRAM load)...")
    # Reuse the pipeline's already-loaded Mistral instance as RAGAS judge
    # This prevents a second LlamaCpp from loading and causing OOM on RTX 3060 6GB
    wrapped_llm = LangchainLLMWrapper(pipeline.llm)
    faithfulness.llm = wrapped_llm

    questions  = []
    answers    = []
    contexts   = []

    count = 0
    print(f"Collecting pipeline outputs (target: {N_EVAL} non-crisis prompts)...")
    for prompt in prompts:
        if count >= N_EVAL:
            break
        result = pipeline.run(prompt["text"])
        if result["crisis"]:
            continue   # skip crisis intercepts — no retrieved context to evaluate
        questions.append(prompt["text"])
        answers.append(result["response"])
        contexts.append(result["retrieved_chunks"])
        count += 1
        print(f"  [{count:02d}/{N_EVAL}] {prompt['emotion']:<12} | {prompt['text'][:50]}...")

    print(f"\nBuilding RAGAS dataset ({len(questions)} samples)...")
    ds = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    print("Running RAGAS faithfulness evaluation (local Mistral judge)...")
    result = ragas_evaluate(ds, metrics=[faithfulness])
    score  = result["faithfulness"]

    print(f"\nRAGAS Faithfulness: {score:.4f}  (target: > 0.65)")
    print("PASS" if score >= 0.65 else "BELOW TARGET (target 0.65)")

    output = {
        "faithfulness": round(float(score), 4),
        "target":       0.65,
        "pass":         float(score) >= 0.65,
        "n_evaluated":  len(questions),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_ragas_eval()
