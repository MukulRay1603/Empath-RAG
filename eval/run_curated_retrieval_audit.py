"""
Audit curated-support retrieval without running the generator.

Run after Karthik's corpus has been validated and indexed:
    python eval/run_curated_retrieval_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pipeline.pipeline import EmpathRAGPipeline
from pipeline.query_router import route_query


PROMPTS = [
    "I'm so anxious about my thesis defense next week, I can't sleep.",
    "My advisor rejected my work again and I don't know what to do.",
    "I feel isolated in my program and I don't have anyone to talk to.",
    "I think I might need counseling but I'm not sure where to start.",
    "I'm burned out and falling behind on everything.",
    "Can you help me find support for disability accommodations?",
    "I don't know who to contact after hours if things get worse.",
]

RESULTS_PATH = Path("eval/curated_retrieval_audit.json")


def main() -> int:
    pipeline = EmpathRAGPipeline(
        retrieval_corpus="curated_support",
        use_real_guardrail=True,
        guardrail_threshold=0.5,
    )

    rows = []
    for prompt in PROMPTS:
        emotion = pipeline._classify_emotion(prompt)
        pipeline.tracker.update(emotion, len(prompt.split()))
        trajectory = pipeline.tracker.trajectory()
        routed = route_query(prompt, emotion, trajectory)
        retrieved = pipeline._retrieve(routed, emotion)
        sources = pipeline._source_summaries(retrieved)
        rows.append(
            {
                "prompt": prompt,
                "emotion": emotion,
                "trajectory": trajectory,
                "routed_query": routed,
                "sources": sources,
            }
        )

        print("\nPROMPT:", prompt)
        if not sources:
            print("  NO SOURCES")
            continue
        for i, source in enumerate(sources[:3], 1):
            print(
                f"  {i}. {source['source_name']} | {source['title']} | "
                f"{source['topic']} | {source['risk_level']}"
            )
            if source["url"]:
                print(f"     {source['url']}")

    RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved audit: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
