"""Free local detectors, used as reference points next to Sapling.

Every scorer returns values where higher means more AI-like. The models only
ever read text here; none of them generates any.
"""

import logging

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

log = logging.getLogger(__name__)

# name -> (hub id, the label that means "AI")
CLASSIFIERS = {
    # OpenAI's 2019 detector, trained on GPT-2 output. Kept as a dated baseline.
    "roberta_openai": ("openai-community/roberta-base-openai-detector", "Fake"),
    # A 2025 RoBERTa detector trained on output from recent chat models.
    "roberta_fakespot": ("fakespot-ai/roberta-base-ai-text-detection-v1", "AI"),
}
# Fast-DetectGPT's own choice of scoring model for the black-box setting.
ZERO_SHOT_MODEL = "EleutherAI/gpt-neo-2.7B"
CLASSIFIER_MAX_TOKENS = 512
ZERO_SHOT_MAX_TOKENS = 1024
BATCH_SIZE = 16


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ai_probability(hub_id: str, ai_label: str, texts: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(hub_id).to(_device()).eval()
    ai_index = model.config.label2id[ai_label]
    # The Fakespot card asks for single-line input; harmless for the other model.
    flat = [" ".join(text.split()) for text in texts]
    probabilities: list[float] = []
    for start in range(0, len(flat), BATCH_SIZE):
        batch = tokenizer(
            flat[start:start + BATCH_SIZE], truncation=True, padding=True,
            max_length=CLASSIFIER_MAX_TOKENS, return_tensors="pt",
        ).to(_device())
        with torch.inference_mode():
            logits = model(**batch).logits
        probabilities += logits.softmax(-1)[:, ai_index].tolist()
    return np.array(probabilities)


def sampling_discrepancy(logits: torch.Tensor, token_ids: torch.Tensor) -> tuple[float, float]:
    """Fast-DetectGPT's analytic curvature (Bao et al., 2024) and the mean token
    log-likelihood, from next-token logits [T, V] and the tokens that came next [T].

    The curvature asks how much more likely the actual text is than what the
    model would expect of text sampled from its own next-token distributions,
    in units of that expectation's standard deviation.
    """
    log_probs = logits.float().log_softmax(-1)
    probs = log_probs.exp()
    log_likelihood = log_probs.gather(-1, token_ids[:, None]).squeeze(-1)
    expected = (probs * log_probs).sum(-1)
    variance = (probs * log_probs.square()).sum(-1) - expected.square()
    curvature = (log_likelihood.sum() - expected.sum()) / variance.sum().sqrt()
    return curvature.item(), log_likelihood.mean().item()


def zero_shot_scores(texts: list[str]) -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL)
    dtype = torch.float16 if _device() == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ZERO_SHOT_MODEL, torch_dtype=dtype)
    model = model.to(_device()).eval()
    curvatures, log_likelihoods = [], []
    for text in texts:
        ids = tokenizer(
            text, truncation=True, max_length=ZERO_SHOT_MAX_TOKENS, return_tensors="pt"
        ).input_ids.to(_device())
        with torch.inference_mode():
            logits = model(ids).logits[0, :-1]
        curvature, log_likelihood = sampling_discrepancy(logits, ids[0, 1:])
        curvatures.append(curvature)
        log_likelihoods.append(log_likelihood)
    return {
        "fast_detectgpt": np.array(curvatures),
        # The classic perplexity detector: text the model finds predictable is suspect.
        "log_likelihood": np.array(log_likelihoods),
    }


def classifier_scores(texts: list[str]) -> dict[str, np.ndarray]:
    scores = {}
    for name, (hub_id, ai_label) in CLASSIFIERS.items():
        log.info("scoring %d texts with %s", len(texts), hub_id)
        scores[name] = _ai_probability(hub_id, ai_label, texts)
    return scores


def score_all(texts: list[str]) -> dict[str, np.ndarray]:
    scores = classifier_scores(texts)
    log.info("scoring %d texts with %s", len(texts), ZERO_SHOT_MODEL)
    return scores | zero_shot_scores(texts)
