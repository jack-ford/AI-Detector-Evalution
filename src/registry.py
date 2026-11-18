"""What the detectors are called and where they come from. No model code lives here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    label: str
    family: str
    year: int
    models: tuple[str, ...]
    license: str
    outputs_probability: bool


SUPERVISED, ZERO_SHOT = "supervised", "zero-shot"

DETECTORS = (
    DetectorInfo("desklib", "Desklib DeBERTa-v3-large", SUPERVISED, 2025,
                 ("desklib/ai-text-detector-v1.01",), "MIT", True),
    DetectorInfo("roberta_fakespot", "Fakespot RoBERTa-base", SUPERVISED, 2025,
                 ("fakespot-ai/roberta-base-ai-text-detection-v1",), "Apache-2.0", True),
    DetectorInfo("radar", "RADAR RoBERTa-large", SUPERVISED, 2023,
                 ("TrustSafeAI/RADAR-Vicuna-7B",), "non-commercial", True),
    DetectorInfo("chatgpt_roberta", "HC3 RoBERTa-base", SUPERVISED, 2023,
                 ("Hello-SimpleAI/chatgpt-detector-roberta",), "not stated", True),
    DetectorInfo("openai_large", "OpenAI RoBERTa-large (GPT-2)", SUPERVISED, 2019,
                 ("openai-community/roberta-large-openai-detector",), "MIT", True),
    DetectorInfo("openai_base", "OpenAI RoBERTa-base (GPT-2)", SUPERVISED, 2019,
                 ("openai-community/roberta-base-openai-detector",), "MIT", True),
    DetectorInfo("binoculars", "Binoculars (Qwen2.5-1.5B)", ZERO_SHOT, 2024,
                 ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"), "Apache-2.0", False),
    DetectorInfo("fast_detectgpt", "Fast-DetectGPT (GPT-Neo-2.7B)", ZERO_SHOT, 2024,
                 ("EleutherAI/gpt-neo-2.7B",), "MIT", False),
    DetectorInfo("log_likelihood", "Log-likelihood (GPT-Neo-2.7B)", ZERO_SHOT, 2019,
                 ("EleutherAI/gpt-neo-2.7B",), "MIT", False),
)

BY_NAME = {d.name: d for d in DETECTORS}
NAMES = [d.name for d in DETECTORS]
PROBABILITY_NAMES = {d.name for d in DETECTORS if d.outputs_probability}


def label(name: str) -> str:
    return BY_NAME[name].label
