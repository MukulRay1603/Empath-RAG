"""
smoke_test_pipeline.py
Run from repo root: python smoke_test_pipeline.py

Tests pipeline.run() on 5 inputs — one per emotion class.
Prints per-stage latency, retrieved chunk preview, response preview.
Does NOT require DeBERTa checkpoint — stub guardrail is used.
"""

import sys, json, textwrap
sys.path.insert(0, "src")

from pipeline.pipeline import EmpathRAGPipeline

LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]

TEST_INPUTS = [
    {"text": "I feel completely hopeless and I don't see a point anymore.",
     "expected_emotion": "distress"},
    {"text": "I'm so anxious about my thesis defense next week, I can't sleep.",
     "expected_emotion": "anxiety"},
    {"text": "My advisor rejected my work again without even reading it properly.",
     "expected_emotion": "frustration"},
    {"text": "Can you give me some tips on how to structure a literature review?",
     "expected_emotion": "neutral"},
    {"text": "I finally finished my dissertation chapter and my advisor loved it!",
     "expected_emotion": "hopeful"},
]

def fmt_latency(lat: dict) -> str:
    parts = [f"{k.replace('_ms','')}={v}ms" for k, v in lat.items() if k != "total_ms"]
    return f"[{' | '.join(parts)} | total={lat.get('total_ms',0)}ms]"

def run_smoke_test():
    print("=" * 70)
    print("EmpathRAG Smoke Test")
    print("=" * 70)

    print("\nInitialising pipeline (this takes ~10s)...")
    pipeline = EmpathRAGPipeline(use_real_guardrail=False)

    passed = 0
    failed = 0

    for i, test in enumerate(TEST_INPUTS):
        print(f"\n{'─'*70}")
        print(f"Test {i+1}/5 — expected emotion: {test['expected_emotion']}")
        print(f"Input: {test['text']}")

        result = pipeline.run(test["text"])

        emotion_name = result["emotion_name"]
        trajectory   = result["trajectory"]
        crisis       = result["crisis"]
        chunks       = result["retrieved_chunks"]
        response     = result["response"]
        latency      = result["latency_ms"]

        # Verify
        emotion_ok = (emotion_name == test["expected_emotion"])
        chunks_ok  = len(chunks) > 0
        response_ok= len(response) > 20

        status = "PASS" if (emotion_ok and chunks_ok and response_ok) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\nStatus     : {status}")
        print(f"Emotion    : {emotion_name} (expected: {test['expected_emotion']}) "
              f"{'✓' if emotion_ok else '✗ MISMATCH'}")
        print(f"Trajectory : {trajectory}")
        print(f"Crisis     : {crisis}")
        print(f"Chunks     : {len(chunks)} retrieved {'✓' if chunks_ok else '✗ NONE'}")
        if chunks:
            preview = chunks[0][:120].replace("\n", " ")
            print(f"Top chunk  : {preview}...")
        print(f"Response   : {response[:150].replace(chr(10), ' ')}...")
        print(f"Latency    : {fmt_latency(latency)}")

    print(f"\n{'='*70}")
    print(f"Results: {passed}/5 passed, {failed}/5 failed")

    if failed == 0:
        print("✅ All smoke tests passed. Pipeline is working end-to-end.")
        print("\nNext step: once DeBERTa checkpoint lands in models/safety_guardrail/,")
        print("re-run with use_real_guardrail=True to verify guardrail intercepts.")
    else:
        print("⚠️  Some tests failed. Check emotion predictions above.")
        print("   If emotion mismatches — RoBERTa checkpoint may not be loaded correctly.")
        print("   If no chunks — verify FAISS index path and SQLite annotation.")

    # Save results to file
    results_summary = {
        "passed": passed,
        "failed": failed,
        "per_test": [
            {
                "input": t["text"],
                "expected": t["expected_emotion"],
                "got": LABEL_NAMES[pipeline._classify_emotion(t["text"])],
            }
            for t in TEST_INPUTS
        ]
    }
    with open("eval/smoke_test_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults saved to eval/smoke_test_results.json")


if __name__ == "__main__":
    run_smoke_test()
