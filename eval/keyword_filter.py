import json


CRISIS_KEYWORDS = [
    "kill myself",
    "end my life",
    "commit suicide",
    "want to die",
    "take my life",
    "suicidal",
    "overdose",
    "no reason to live",
    "end it all",
    "goodbye forever",
    "don't want to be alive",
    "hurt myself",
    "self harm",
    "cutting myself",
    "end things",
    "not be here",
    "better off without me",
    "goodbye notes",
]


def keyword_filter(text: str) -> bool:
    """Returns True if any crisis keyword is found in the text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)


def evaluate_probes(probes_path="eval/adversarial_probes.json"):
    import pandas as pd

    with open(probes_path) as f:
        probes = json.load(f)

    results = []
    for p in probes:
        fired = keyword_filter(p["text"])
        results.append(
            {
                "id": p["id"],
                "category": p["category"],
                "keyword_fired": fired,
                "text": p["text"],
            }
        )

    df = pd.DataFrame(results)
    print("\nKeyword filter results by category:")
    summary = df.groupby("category")["keyword_fired"].agg(["sum", "count", "mean"]).round(2)
    summary.columns = ["intercepted", "total", "intercept_rate"]
    print(summary)
    return df


if __name__ == "__main__":
    evaluate_probes()
