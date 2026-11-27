"""Gemma-based zero-shot scorer replacing ThalamusDB's CLIP / Sentence-BERT / audio models.

For an item (text, image, or audio clip) and a natural-language predicate, the score is
    logit("Yes") - logit("No")
at the first answer position, from a single forward pass. Higher means the model believes
the predicate holds more strongly. ThalamusDB only needs a monotone similarity score plus
a threshold, so this plugs straight into its threshold-calibration design. The model's own
decision boundary (score = 0) serves as the ground-truth threshold for simulated labels.
"""

import time

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, prepare_inputs

MAX_TEXT_CHARS = 2000  # Long YouTube descriptions are truncated to bound per-item cost.

PROMPTS = {
    "image": 'Does the following text describe the image? "{pred}"\nAnswer Yes or No.',
    "audio": 'Does the following text describe the audio clip? "{pred}"\nAnswer Yes or No.',
    "text": (
        'Does the following text match the description "{pred}"?\n'
        "Text: {item}\nAnswer Yes or No."
    ),
}


class GemmaScorer:
    def __init__(self, model_path, modality):
        self.modality = modality
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]

    def _prompt(self, pred, item):
        if self.modality == "text":
            text = str(item)[:MAX_TEXT_CHARS]
            raw = PROMPTS["text"].format(pred=pred, item=text)
            return apply_chat_template(self.processor, self.config, raw, num_images=0)
        raw = PROMPTS[self.modality].format(pred=pred)
        if self.modality == "image":
            return apply_chat_template(self.processor, self.config, raw, num_images=1)
        return apply_chat_template(self.processor, self.config, raw, num_images=0, num_audios=1)

    def score(self, pred, item):
        """Return (score, seconds). item is text, or a path to an image / audio file."""
        start = time.perf_counter()
        prompt = self._prompt(pred, item)
        kwargs = {}
        if self.modality == "image":
            kwargs["images"] = [item]
        elif self.modality == "audio":
            kwargs["audio"] = [item]
        inputs = prepare_inputs(
            self.processor,
            prompts=[prompt],
            image_token_index=getattr(self.model.config, "image_token_index", None),
            **kwargs,
        )
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
        # Same path as mlx-vlm generation. Calling self.model(...) directly passes a dense
        # (non-causal) 4D mask for multimodal inputs, which corrupts the logits.
        emb = self.model.get_input_embeddings(
            inputs["input_ids"], inputs.get("pixel_values"), mask=inputs.get("attention_mask"), **extra
        )
        lm_kwargs = {k: v for k, v in emb.to_dict().items() if k not in ("inputs_embeds", "attention_mask_4d") and v is not None}
        out = self.model.language_model(inputs["input_ids"], inputs_embeds=emb.inputs_embeds, **lm_kwargs)
        logits = out.logits if hasattr(out, "logits") else out
        last = logits[0, -1].astype(mx.float32)
        value = (last[self.yes_id] - last[self.no_id]).item()
        return value, time.perf_counter() - start
