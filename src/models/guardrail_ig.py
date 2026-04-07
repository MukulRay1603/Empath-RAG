import torch
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CHECKPOINT = "models/safety_guardrail"
CRISIS_LABEL = 0
HYPOTHESIS = "This person is expressing suicidal ideation or intent to self-harm."


class SafetyGuardrail:
    def __init__(self, checkpoint=CHECKPOINT):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
        self.ig = IntegratedGradients(self._forward)

    def _forward(self, input_ids, attention_mask=None, token_type_ids=None):
        emb = self.model.deberta.embeddings.word_embeddings(input_ids)
        return self.model(inputs_embeds=emb, attention_mask=attention_mask).logits

    def check(self, text: str, threshold: float = 0.5):
        """Returns (is_crisis: bool, confidence: float, token_attributions: list)."""
        enc = self.tokenizer(
            text, HYPOTHESIS, return_tensors="pt", truncation=True, max_length=256
        )
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        crisis_prob = probs[0, CRISIS_LABEL].item()

        if crisis_prob < threshold:
            return False, crisis_prob, []

        input_ids = enc["input_ids"]
        baseline = torch.zeros_like(input_ids)
        attrs, _ = self.ig.attribute(
            input_ids,
            baselines=baseline,
            target=CRISIS_LABEL,
            additional_forward_args=(
                enc.get("attention_mask"),
                enc.get("token_type_ids"),
            ),
            return_convergence_delta=True,
            n_steps=50,
        )
        token_attrs = attrs.sum(-1).abs().squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        paired = sorted(zip(tokens, token_attrs), key=lambda x: x[1], reverse=True)[:5]
        return True, crisis_prob, paired
