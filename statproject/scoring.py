"""Run the detectors over the corpus and the essay-rewrite chunks."""

import json
import logging

import pandas as pd
import torch

from . import paths, registry
from .corpus import load_metadata, load_rewrite_chunks, load_texts
from .detectors import score_detector

log = logging.getLogger(__name__)

MODELS_FILE = "models.json"


def _write_scores(name: str, ids: pd.Series, scores) -> None:
    out = paths.SCORES_DIR / f"{name}.csv"
    pd.DataFrame({"id": ids, "score": scores}).to_csv(out, index=False)
    log.info("wrote %s", out)


def _remember_revisions(name: str, revisions: dict[str, str]) -> None:
    path = paths.SCORES_DIR / MODELS_FILE
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known[name] = revisions
    path.write_text(json.dumps(known, indent=2, sort_keys=True), encoding="utf-8")


def score_everything(only: list[str] | None = None) -> None:
    metadata = load_metadata()
    chunks = load_rewrite_chunks()
    ids = pd.concat([metadata["id"], chunks["id"]], ignore_index=True)
    texts = pd.concat([load_texts(metadata["id"]), chunks["text"]], ignore_index=True).tolist()

    paths.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for name in only or registry.NAMES:
        if name in done:
            continue
        log.info("scoring %d texts with %s", len(texts), name)
        scores, revisions = score_detector(name, texts)
        for produced, values in scores.items():
            _write_scores(produced, ids, values)
            _remember_revisions(produced, revisions)
        done.update(scores)
        torch.cuda.empty_cache()
