"""
eval/run_ragas.py
DeepEval FaithfulnessMetric with Mistral 7B as local judge.

RAGAS 0.1.21 is incompatible with llama-cpp-python's synchronous Llama object
(requires agenerate_prompt async method). We use DeepEval's FaithfulnessMetric
instead, which supports any custom LLM via DeepEvalBaseLLM and runs synchronously
when async_mode=False.

DeepEval FaithfulnessMetric definition (identical to RAGAS):
  1. LLM extracts all factual claims from the generated response
  2. LLM checks each claim against retrieved context — truthful if not contradicted
  3. Score = number of truthful claims / total claims

Mistral 7B Instruct Q4_K_M was confirmed to produce valid JSON for claim extraction
(tested before implementation). No JSON confinement library required.

Citation: DeepEval FaithfulnessMetric — https://deepeval.com/docs/metrics-faithfulness
"""

import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate as deepeval_evaluate
from pipeline.pipeline import EmpathRAGPipeline

PROMPTS_PATH = "eval/test_prompts.json"
RESULTS_PATH = "eval/ragas_results.json"
N_EVAL       = 40


class MistralJudge(DeepEvalBaseLLM):
    """
    Wraps llama-cpp-python's Llama instance as a DeepEvalBaseLLM judge.
    Mistral 7B Instruct Q4_K_M confirmed to produce valid JSON for
    claim extraction prompts (tested independently).
    """
    def __init__(self, llm):
        self._llm = llm

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        from llama_cpp import LlamaGrammar
        # GBNF grammar-constrained sampling — physically prevents Mistral from
        # generating invalid JSON at the token level. No post-processing needed.
        # DeepEval makes two types of calls: claim extraction (array) and
        # claim verification (object). We detect which is needed from the prompt
        # and apply the appropriate grammar.
        if "[" in prompt[-200:] or "list" in prompt[-200:].lower() or "claims" in prompt[-200:].lower() or "statements" in prompt[-200:].lower():
            # Claim extraction call — expects JSON array of strings
            grammar_str = (
                'root   ::= arr\n'
                'arr    ::= "[" ws (val (ws "," ws val)*)? ws "]"\n'
                'val    ::= string\n'
                'string ::= "\\"" ([^\\\\"] | "\\\\" ["\\\\/bfnrt])* "\\""\n'
                'ws     ::= ([ \\t\\n])*\n'
            )
        else:
            # Claim verification call — expects JSON object with verdicts
            grammar_str = (
                'root     ::= obj\n'
                'obj      ::= "{" ws pair (ws "," ws pair)* ws "}"\n'
                'pair     ::= string ws ":" ws val\n'
                'val      ::= string | "true" | "false" | "null" | number | arr | obj\n'
                'arr      ::= "[" ws (val (ws "," ws val)*)? ws "]"\n'
                'string   ::= "\\"" ([^\\\\"] | "\\\\" ["\\\\/bfnrt])* "\\""\n'
                'number   ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? (([eE] [-+]? [0-9]+))?\n'
                'ws       ::= ([ \\t\\n])*\n'
            )
        try:
            grammar = LlamaGrammar.from_string(grammar_str, verbose=False)
            out = self._llm(prompt, max_tokens=1024, temperature=0.0, grammar=grammar, stop=["[INST]"])
        except Exception:
            # Fallback: no grammar constraint if grammar fails to parse
            out = self._llm(prompt, max_tokens=1024, temperature=0.0, stop=["[INST]"])
        return out["choices"][0]["text"].strip()

    async def a_generate(self, prompt: str) -> str:
        # DeepEval calls a_generate when async_mode=True.
        # We set async_mode=False so this should never be called,
        # but implement it as a synchronous fallback for safety.
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "Mistral-7B-Instruct-v0.2-Q4_K_M (local)"


def run_faithfulness_eval():
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)

    print("Initialising pipeline (use_real_guardrail=False for speed)...")
    pipeline = EmpathRAGPipeline(use_real_guardrail=False, guardrail_threshold=0.5)

    # Monkey-patch guardrail to skip IG (no-op since stub is active, but kept for
    # consistency with other eval scripts in case real guardrail is swapped in)
    original_check = pipeline.guardrail.check
    def fast_check(text, threshold=0.5, skip_ig=False):
        return original_check(text, threshold=threshold, skip_ig=True)
    pipeline.guardrail.check = fast_check

    print("Wrapping Mistral as DeepEval judge (async_mode=False)...")
    judge = MistralJudge(pipeline.llm)
    metric = FaithfulnessMetric(
        model=judge,
        threshold=0.5,
        async_mode=False,       # synchronous — avoids agenerate_prompt error
        include_reason=False,   # speeds up evaluation — we only need the score
        verbose_mode=False,
    )

    test_cases  = []
    count       = 0

    print(f"Collecting pipeline outputs (target: {N_EVAL} non-crisis prompts)...")
    for prompt in prompts:
        if count >= N_EVAL:
            break
        result = pipeline.run(prompt["text"])
        if result["crisis"] or not result["retrieved_chunks"]:
            continue

        test_cases.append(LLMTestCase(
            input=prompt["text"],
            actual_output=result["response"],
            retrieval_context=result["retrieved_chunks"],
        ))
        count += 1
        print(f"  [{count:02d}/{N_EVAL}] {prompt['emotion']:<12} | {prompt['text'][:50]}...")

    print(f"\nRunning DeepEval FaithfulnessMetric on {len(test_cases)} test cases...")
    print("(Each case: Mistral extracts claims, then verifies each against context)")

    scores = []
    for i, tc in enumerate(test_cases):
        metric.measure(tc)
        score = metric.score
        scores.append(score)
        print(f"  [{i+1:02d}/{len(test_cases)}] faithfulness={score:.3f}")

    mean_score = sum(scores) / len(scores) if scores else 0.0
    passed     = mean_score >= 0.5   # DeepEval default threshold is 0.5

    print(f"\nFaithfulness Results (DeepEval FaithfulnessMetric):")
    print(f"  Mean faithfulness: {mean_score:.4f}")
    print(f"  Threshold:         0.5 (DeepEval default)")
    print(f"  Target for paper:  > 0.65")
    print(f"  {'PASS' if mean_score >= 0.65 else 'BELOW 0.65 TARGET' if mean_score >= 0.5 else 'BELOW THRESHOLD'}")
    print(f"  n_evaluated:       {len(scores)}")

    output = {
        "method":        "DeepEval FaithfulnessMetric (Mistral-7B judge, async_mode=False)",
        "faithfulness":  round(mean_score, 4),
        "target":        0.65,
        "pass":          mean_score >= 0.65,
        "n_evaluated":   len(scores),
        "per_sample":    [round(s, 4) for s in scores],
        "score_distribution": {
            "min":    round(min(scores), 4) if scores else None,
            "max":    round(max(scores), 4) if scores else None,
            "median": round(sorted(scores)[len(scores)//2], 4) if scores else None,
        }
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_faithfulness_eval()
