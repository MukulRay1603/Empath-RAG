import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm

CHECKPOINT = "models/emotion_classifier"
DB_PATH = "data/indexes/metadata.db"
SAFETY_SCORE_MAP = {0: 0.0, 1: 0.0, 2: 0.3, 3: 0.7, 4: 1.0}
BATCH = 128


def annotate_corpus(checkpoint=CHECKPOINT, db_path=DB_PATH):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    model = PeftModel.from_pretrained(base, checkpoint).eval()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE emotion_label = -1"
    ).fetchall()
    print(f"Chunks to annotate: {len(rows)}")

    for i in tqdm(range(0, len(rows), BATCH), desc="Annotating"):
        batch = rows[i : i + BATCH]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        enc = tokenizer(
            texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**enc).logits
        labels = logits.argmax(-1).tolist()
        for rid, lbl in zip(ids, labels):
            score = SAFETY_SCORE_MAP[lbl]
            conn.execute(
                "UPDATE chunks SET emotion_label=?, safety_score=? WHERE id=?",
                (lbl, score, rid),
            )
        conn.commit()

    conn.close()
    print("Annotation complete.")


if __name__ == "__main__":
    annotate_corpus()
