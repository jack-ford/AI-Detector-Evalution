"""Reading and writing the passage corpus and the essay-rewrite set."""

import csv
import logging
from pathlib import Path

import pandas as pd

from . import paths
from .text import normalize, word_count

log = logging.getLogger(__name__)

METADATA_COLUMNS = [
    "id", "label", "subset", "genre", "prompt_style", "n_words", "source", "license",
    "title", "url", "date", "cluster", "pair_id", "writer", "detail",
]
# The rewrite set has each of five essays in four versions, named by the share
# of AI involvement the authors intended; 25 and 75 are hybrids with no clean label.
REWRITE_LABELS = {0: "human", 25: "mixed", 75: "mixed", 100: "ai"}
CHUNK_WORDS = 100
MIN_CHUNK_WORDS = 80
MAX_CHUNKS = 6


def passage_path(passage_id: str) -> Path:
    folder = paths.HUMAN_DIR if passage_id.startswith("h-") else paths.AI_DIR
    return folder / f"{passage_id}.txt"


def read_passage(passage_id: str) -> str:
    return passage_path(passage_id).read_text(encoding="utf-8")


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(paths.METADATA, dtype=str, keep_default_na=False).astype({"n_words": int})


def write_metadata(rows: list[dict]) -> None:
    with paths.METADATA.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def load_texts(ids: pd.Series) -> pd.Series:
    return pd.Series([read_passage(i) for i in ids], index=ids.index)


def build_rewrite_chunks() -> pd.DataFrame:
    """Cut each essay version into consecutive 100-word chunks, six at most."""
    rows = []
    for path in sorted(paths.REWRITE_ESSAYS.glob("*.txt")):
        essay, percent_ai = (int(part) for part in path.stem.split("-"))
        words = normalize(path.read_text(encoding="utf-8")).split()
        for k in range(MAX_CHUNKS):
            chunk = words[k * CHUNK_WORDS:(k + 1) * CHUNK_WORDS]
            if len(chunk) < MIN_CHUNK_WORDS:
                break
            rows.append({
                "id": f"{essay}-{percent_ai}-{k + 1}",
                "essay": essay,
                "percent_ai": percent_ai,
                "chunk": k + 1,
                "label": REWRITE_LABELS[percent_ai],
                "n_words": len(chunk),
                "text": " ".join(chunk),
            })
    chunks = pd.DataFrame(rows)
    chunks.to_csv(paths.REWRITE_CHUNKS, index=False)
    log.info("wrote %d essay chunks to %s", len(chunks), paths.REWRITE_CHUNKS)
    return chunks


def load_rewrite_chunks() -> pd.DataFrame:
    return pd.read_csv(paths.REWRITE_CHUNKS)


def load_scores() -> pd.DataFrame:
    """One row per scored text, one column per local detector."""
    frames = [
        pd.read_csv(path, index_col="id")["score"].rename(path.stem)
        for path in sorted(paths.SCORES_DIR.glob("*.csv"))
    ]
    return pd.concat(frames, axis=1)
