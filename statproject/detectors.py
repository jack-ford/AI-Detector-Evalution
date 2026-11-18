"""Open-source detectors, run locally. They only ever read text; none writes any.

Every scorer returns one number per text where higher means more AI-like.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

log = logging.getLogger(__name__)

Scores = dict[str, np.ndarray]
Revisions = dict[str, str]

CLASSIFIER_MAX_TOKENS = 512
DESKLIB_MAX_TOKENS = 768
BINOCULARS_MAX_TOKENS = 512
ZERO_SHOT_MAX_TOKENS = 1024
BATCH_SIZE = 16


@dataclass(frozen=True)
class Classifier:
    hub_id: str
    # A label name looked up in the model's config, or a raw output index.
    ai_class: str | int
    # The Fakespot card asks for single-line input.
    single_line: bool = False


CLASSIFIERS = {
    "roberta_fakespot": Classifier("fakespot-ai/roberta-base-ai-text-detection-v1", "AI", single_line=True),
    "radar": Classifier("TrustSafeAI/RADAR-Vicuna-7B", 0),
    "chatgpt_roberta": Classifier("Hello-SimpleAI/chatgpt-detector-roberta", "ChatGPT"),
    # The large model's config has no label names; OpenAI's own server code unpacks
    # its output as (fake, real), so index 0 is AI.
    "openai_large": Classifier("openai-community/roberta-large-openai-detector", 0),
    "openai_base": Classifier("openai-community/roberta-base-openai-detector", "Fake"),
}
DESKLIB = "desklib/ai-text-detector-v1.01"
BINOCULARS_OBSERVER = "Qwen/Qwen2.5-1.5B"
BINOCULARS_PERFORMER = "Qwen/Qwen2.5-1.5B-Instruct"
ZERO_SHOT_MODEL = "EleutherAI/gpt-neo-2.7B"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _revision(hub_id: str) -> str:
    """The commit the local copy of a model was downloaded at."""
    return Path(snapshot_download(hub_id, local_files_only=True)).name


def _batches(texts: list[str]):
    for start in range(0, len(texts), BATCH_SIZE):
        yield texts[start:start + BATCH_SIZE]


def classifier_scores(spec: Classifier, texts: list[str]) -> tuple[np.ndarray, Revisions]:
    tokenizer = AutoTokenizer.from_pretrained(spec.hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hub_id).to(_device()).eval()
    ai_index = model.config.label2id[spec.ai_class] if isinstance(spec.ai_class, str) else spec.ai_class
    if spec.single_line:
        texts = [" ".join(text.split()) for text in texts]
    probabilities: list[float] = []
    for batch in _batches(texts):
        encoded = tokenizer(
            batch, truncation=True, padding=True, max_length=CLASSIFIER_MAX_TOKENS, return_tensors="pt"
        ).to(_device())
        with torch.inference_mode():
            logits = model(**encoded).logits
        probabilities += logits.softmax(-1)[:, ai_index].tolist()
    return np.array(probabilities), {spec.hub_id: _revision(spec.hub_id)}


class DesklibModel(PreTrainedModel):
    """The architecture from the Desklib model card: DeBERTa, mean pooling, one logit."""

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled).squeeze(-1)


def desklib_scores(texts: list[str]) -> tuple[np.ndarray, Revisions]:
    tokenizer = AutoTokenizer.from_pretrained(DESKLIB)
    model = DesklibModel.from_pretrained(DESKLIB).to(_device()).eval()
    probabilities: list[float] = []
    for batch in _batches(texts):
        encoded = tokenizer(
            batch, truncation=True, padding=True, max_length=DESKLIB_MAX_TOKENS, return_tensors="pt",
            return_token_type_ids=False,
        ).to(_device())
        with torch.inference_mode():
            probabilities += model(**encoded).sigmoid().tolist()
    return np.array(probabilities), {DESKLIB: _revision(DESKLIB)}


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


def binoculars_ratio(observer_logits: torch.Tensor, performer_logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """Binoculars (Hans et al., 2024): the performer's log-perplexity of the text
    divided by the cross-perplexity between the two models' predictions. Low
    values mean machine-written, so callers negate it.

    Logits are [T, V] for the next-token predictions, token_ids is [T].
    """
    performer_log_probs = performer_logits.float().log_softmax(-1)
    log_perplexity = -performer_log_probs.gather(-1, token_ids[:, None]).mean()
    cross_perplexity = -(observer_logits.float().softmax(-1) * performer_log_probs).sum(-1).mean()
    return (log_perplexity / cross_perplexity).item()


def _fp16_if_cuda() -> torch.dtype:
    return torch.float16 if _device() == "cuda" else torch.float32


def binoculars_scores(texts: list[str]) -> tuple[np.ndarray, Revisions]:
    tokenizer = AutoTokenizer.from_pretrained(BINOCULARS_OBSERVER)
    observer = AutoModelForCausalLM.from_pretrained(BINOCULARS_OBSERVER, dtype=_fp16_if_cuda()).to(_device()).eval()
    performer = AutoModelForCausalLM.from_pretrained(BINOCULARS_PERFORMER, dtype=_fp16_if_cuda()).to(_device()).eval()
    scores = []
    for text in texts:
        ids = tokenizer(
            text, truncation=True, max_length=BINOCULARS_MAX_TOKENS, return_tensors="pt"
        ).input_ids.to(_device())
        with torch.inference_mode():
            ratio = binoculars_ratio(observer(ids).logits[0, :-1], performer(ids).logits[0, :-1], ids[0, 1:])
        scores.append(-ratio)
    revisions = {hub_id: _revision(hub_id) for hub_id in (BINOCULARS_OBSERVER, BINOCULARS_PERFORMER)}
    return np.array(scores), revisions


def zero_shot_scores(texts: list[str]) -> tuple[Scores, Revisions]:
    tokenizer = AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(ZERO_SHOT_MODEL, dtype=_fp16_if_cuda()).to(_device()).eval()
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
    scores = {
        "fast_detectgpt": np.array(curvatures),
        # The classic perplexity detector: text the model finds predictable is suspect.
        "log_likelihood": np.array(log_likelihoods),
    }
    return scores, {ZERO_SHOT_MODEL: _revision(ZERO_SHOT_MODEL)}


def score_detector(name: str, texts: list[str]) -> tuple[Scores, Revisions]:
    """Scores for one detector name; the zero-shot pair share a model, so asking
    for either returns both."""
    if name in CLASSIFIERS:
        scores, revisions = classifier_scores(CLASSIFIERS[name], texts)
        return {name: scores}, revisions
    if name == "desklib":
        scores, revisions = desklib_scores(texts)
        return {name: scores}, revisions
    if name == "binoculars":
        scores, revisions = binoculars_scores(texts)
        return {name: scores}, revisions
    if name in ("fast_detectgpt", "log_likelihood"):
        return zero_shot_scores(texts)
    raise KeyError(f"unknown detector {name!r}")
