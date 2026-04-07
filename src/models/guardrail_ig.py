"""
src/models/guardrail_ig.py
EmpathRAG — DeBERTa NLI Safety Guardrail with Integrated Gradients

Runs on CPU. Model loaded as float32 (upcast from bf16 checkpoint).
IG operates in embedding space — interpolation is continuous and meaningful.
token_type_ids passed through — required by this tokenizer version.
"""

import torch
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CHECKPOINT  = "models/safety_guardrail"
CRISIS_LABEL = 0   # label 0 = entailment = crisis
HYPOTHESIS  = "This person is expressing suicidal ideation or intent to self-harm."


class SafetyGuardrail:
    def __init__(self, checkpoint: str = CHECKPOINT):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Load as float32 on CPU — bf16 checkpoint upcasts cleanly, CPU inference is stable
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(checkpoint, dtype=torch.float32)
            .eval()
        )
        # IG forward operates on embedding vectors, not integer token IDs
        self.ig = IntegratedGradients(self._forward_embeddings)

    def _forward_embeddings(self, inputs_embeds, attention_mask=None, token_type_ids=None):
        """
        Forward pass accepting embedding tensors instead of token IDs.
        Required by IntegratedGradients — interpolation must happen in
        continuous embedding space, not discrete integer token ID space.
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

    def check(self, text: str, threshold: float = 0.5):
        """
        Run guardrail on a single text string.

        Returns:
            is_crisis   (bool)  — True if crisis probability >= threshold
            confidence  (float) — crisis class probability [0, 1]
            top_tokens  (list)  — [(token_str, attribution_score), ...] top-5
                                  empty list if is_crisis is False
        """
        enc = self.tokenizer(
            text, HYPOTHESIS,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )

        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        crisis_prob = probs[0, CRISIS_LABEL].item()

        if crisis_prob < threshold:
            return False, crisis_prob, []

        # ── Integrated Gradients ───────────────────────────────────────────────
        # Get the embedding layer
        embed_layer = self.model.deberta.embeddings.word_embeddings

        input_ids      = enc["input_ids"]           # [1, seq_len]
        attention_mask = enc["attention_mask"]       # [1, seq_len]
        token_type_ids = enc.get("token_type_ids")  # [1, seq_len] or None

        # Input embeddings and all-zeros baseline in embedding space
        input_embeds   = embed_layer(input_ids)                        # [1, seq_len, hidden]
        baseline_embeds = torch.zeros_like(input_embeds)               # [1, seq_len, hidden]

        attrs, _ = self.ig.attribute(
            input_embeds,
            baselines=baseline_embeds,
            target=CRISIS_LABEL,
            additional_forward_args=(attention_mask, token_type_ids),
            return_convergence_delta=True,
            n_steps=50,
        )
        # attrs: [1, seq_len, hidden_dim] → sum over hidden → [seq_len]
        token_scores = attrs.sum(dim=-1).abs().squeeze(0)  # [seq_len]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Filter out padding tokens ([PAD] = token id 0)
        pairs = [
            (tok, score.item())
            for tok, score, mask in zip(tokens, token_scores, attention_mask[0])
            if mask.item() == 1 and tok not in ("[CLS]", "[SEP]", "▁", "<s>", "</s>")
        ]
        top5 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]

        return True, crisis_prob, top5


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SafetyGuardrail from", CHECKPOINT)
    g = SafetyGuardrail()
    print("Model loaded.\n")

    tests = [
        ("I want to kill myself and I have a plan.",         True),
        ("I think everyone would be better off without me.", True),
        ("I'm really stressed about my thesis deadline.",    False),
        ("this exam is literally killing me lol",            False),
    ]

    all_pass = True
    for text, expect_crisis in tests:
        is_crisis, conf, tokens = g.check(text)
        status = "✅" if is_crisis == expect_crisis else "❌"
        print(f"{status} crisis={is_crisis} ({conf:.3f}) | {text}")
        if is_crisis and tokens:
            print(f"   top tokens: {tokens}")
        if is_crisis != expect_crisis:
            all_pass = False

    print("\n✅ guardrail_ig.py verified." if all_pass else "\n❌ at least one case wrong.")