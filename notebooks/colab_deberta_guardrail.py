# EmpathRAG — DeBERTa NLI Safety Guardrail Fine-Tuning
# Run on Google Colab Pro (A100). Expected time: under 2 hours.
# Target: recall > 0.80, precision > 0.65
#
# SETUP INSTRUCTIONS:
# 1. Upload nli_train.csv, nli_val.csv, nli_test.csv to Colab (or mount Drive)
# 2. Set runtime to A100 GPU
# 3. Run all cells in order

# ── Cell 1: Install ──────────────────────────────────────────────────────────
# !pip install transformers datasets evaluate scikit-learn accelerate -q

# ── Cell 2: Mount Drive ──────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")
# SAVE_DIR = "/content/drive/MyDrive/empathrag/safety_guardrail"
# !mkdir -p {SAVE_DIR}

# ── Cell 3: Training script ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import torch
import evaluate as evaluate_lib
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

SAVE_DIR = "/content/drive/MyDrive/empathrag/safety_guardrail"

train_df = pd.read_csv("nli_train.csv")
val_df = pd.read_csv("nli_val.csv")
test_df = pd.read_csv("nli_test.csv")

MODEL = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        batch["hypothesis"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )


train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

f1 = evaluate_lib.load("f1")
recall = evaluate_lib.load("recall")
precision = evaluate_lib.load("precision")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1.compute(predictions=preds, references=labels, pos_label=0)["f1"],
        "recall": recall.compute(predictions=preds, references=labels, pos_label=0)["recall"],
        "precision": precision.compute(predictions=preds, references=labels, pos_label=0)["precision"],
    }


args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="recall",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

results = trainer.evaluate(test_ds)
print(f"Test recall: {results['eval_recall']:.3f} | precision: {results['eval_precision']:.3f}")
print("Target: recall > 0.80, precision > 0.65")
