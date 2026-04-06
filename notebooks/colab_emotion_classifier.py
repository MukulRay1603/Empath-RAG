# EmpathRAG — RoBERTa Emotion Classifier Fine-Tuning
# Run on Google Colab Pro (A100). Expected time: under 90 minutes.
# Target: weighted F1 > 0.75 on the 5-class taxonomy.
#
# SETUP INSTRUCTIONS:
# 1. Upload this file to Google Colab
# 2. Set runtime to A100 GPU
# 3. Run all cells in order

# ── Cell 1: Install dependencies ────────────────────────────────────────────
# !pip install transformers==4.38.2 datasets peft evaluate scikit-learn accelerate -q

# ── Cell 2: Mount Drive ──────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")
# SAVE_DIR = "/content/drive/MyDrive/empathrag/emotion_classifier"
# !mkdir -p {SAVE_DIR}

# ── Cell 3: Training script ──────────────────────────────────────────────────
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
import torch

SAVE_DIR = "/content/drive/MyDrive/empathrag/emotion_classifier"

LABEL_MAP = {
    "grief": 0, "remorse": 0, "fear": 0, "sadness": 0,
    "nervousness": 1, "confusion": 1, "embarrassment": 1,
    "anger": 2, "annoyance": 2, "disappointment": 2, "disgust": 2,
    "neutral": 3,
    "optimism": 4, "relief": 4, "gratitude": 4, "joy": 4,
    "love": 4, "admiration": 4, "amusement": 4, "approval": 4,
    "caring": 4, "curiosity": 4, "desire": 4, "excitement": 4,
    "pride": 4, "realization": 4, "surprise": 4,
}

raw = load_dataset("google-research-datasets/go_emotions", "simplified")
feature_names = raw["train"].features["labels"].feature.names


def remap(example):
    coarse = 3
    for lid in example["labels"]:
        name = feature_names[lid]
        if name in LABEL_MAP:
            coarse = LABEL_MAP[name]
            break
    return {"label": coarse}


dataset = raw.map(remap)

MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")


tokenized = dataset.map(tokenize, batched=True)

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)

base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=preds, references=labels, average="weighted")


args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Training complete — checkpoint saved to Drive")
