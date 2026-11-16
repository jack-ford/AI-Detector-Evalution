"""Reading and writing the passage corpus and the cached Sapling results."""

import ast
import csv
import logging
from pathlib import Path

import pandas as pd

from . import paths
from .text import word_count

log = logging.getLogger(__name__)

METADATA_COLUMNS = [
    "id", "label", "subset", "genre", "prompt_style", "n_words", "source", "license",
    "title", "url", "date", "cluster", "pair_id", "writer", "detail",
]
# In the original study each essay exists in four versions, named by the share
# of AI involvement the authors intended.
LEGACY_LABELS = {0: "human", 25: "mixed", 75: "mixed", 100: "ai"}


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


def convert_legacy_results() -> pd.DataFrame:
    """Turn the original detection_results files into one csv.

    Those files are Python dict reprs, one per 100-word chunk. The chunk text is
    kept exactly as it was sent to Sapling (the original script stripped
    newlines without inserting spaces, so some sentences run together).
    """
    rows = []
    for path in sorted(paths.LEGACY_SAPLING_RAW.glob("*.txt")):
        record = ast.literal_eval(path.read_text(encoding="utf-8"))
        essay, percent_ai, chunk = (int(part) for part in path.stem.split("-"))
        rows.append({
            "id": record["chunk_id"],
            "essay": essay,
            "percent_ai": percent_ai,
            "chunk": chunk,
            "label": LEGACY_LABELS[percent_ai],
            "n_words": word_count(record["text"]),
            "sapling": record["ai_score"],
            "text": record["text"],
        })
    chunks = pd.DataFrame(rows).sort_values(["essay", "percent_ai", "chunk"])
    chunks.to_csv(paths.LEGACY_CHUNKS, index=False)
    log.info("wrote %d cached Sapling results to %s", len(chunks), paths.LEGACY_CHUNKS)
    return chunks


def load_legacy_chunks() -> pd.DataFrame:
    return pd.read_csv(paths.LEGACY_CHUNKS)


def load_scores() -> pd.DataFrame:
    """One row per scored text, one column per local detector."""
    frames = [
        pd.read_csv(path, index_col="id")["score"].rename(path.stem)
        for path in sorted(paths.SCORES_DIR.glob("*.csv"))
    ]
    return pd.concat(frames, axis=1)
