"""Run the local detectors over the corpus and the original Sapling chunks."""

import logging

import pandas as pd

from . import paths
from .corpus import load_legacy_chunks, load_metadata, load_texts

log = logging.getLogger(__name__)


def score_everything() -> None:
    # Imported here so that commands which don't score never load torch.
    from .detectors import score_all

    metadata = load_metadata()
    legacy = load_legacy_chunks()
    ids = pd.concat([metadata["id"], legacy["id"]], ignore_index=True)
    texts = pd.concat([load_texts(metadata["id"]), legacy["text"]], ignore_index=True)

    paths.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for name, scores in score_all(texts.tolist()).items():
        out = paths.SCORES_DIR / f"{name}.csv"
        pd.DataFrame({"id": ids, "score": scores}).to_csv(out, index=False)
        log.info("wrote %s", out)
