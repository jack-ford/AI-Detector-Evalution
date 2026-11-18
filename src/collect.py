import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from . import paths
from .corpus import write_metadata
from .sources import Document, gutenberg, persuade, stackexchange
from .text import cut_passage, normalize, word_count

log = logging.getLogger(__name__)

SEED = 2026
TARGET_WORDS = (90, 320)
MIN_WORDS = 80
MAX_WORDS = 400
MIN_ANCHOR_WORDS = 30
MIN_LETTER_SHARE = 0.78


@dataclass(frozen=True)
class Recipe:
    genre: str
    documents: Callable[[random.Random], Iterator[Document]]
    count: int
    from_start: bool
    subset: str = "main"


RECIPES = (
    Recipe("forum", stackexchange.answers, 50, from_start=True),
    Recipe("fiction", gutenberg.fiction, 50, from_start=False),
    Recipe("student", persuade.essays, 50, from_start=True),
    # Human-only side set for the false-positive analysis, never paired with AI text.
    Recipe("student", persuade.ell_matched_pairs, 100, from_start=True, subset="ell_probe"),
    # Real student essays that the AI is later asked to imitate one for one.
    Recipe("student", persuade.essays, 30, from_start=True, subset="imitation_probe"),
)


def _is_prose(paragraph: str) -> bool:
    letters = sum(ch.isalpha() or ch.isspace() for ch in paragraph)
    return (
        word_count(paragraph) >= MIN_ANCHOR_WORDS
        and letters / len(paragraph) >= MIN_LETTER_SHARE
        and not paragraph.isupper()
    )


def _excerpt(doc: Document, rng: random.Random, from_start: bool) -> str | None:
    anchors = [i for i, p in enumerate(doc.paragraphs) if _is_prose(p)]
    if not anchors:
        return None
    start = anchors[0] if from_start else rng.choice(anchors)
    passage = normalize(cut_passage(doc.paragraphs, rng.randint(*TARGET_WORDS), start))
    if not MIN_WORDS <= word_count(passage) <= MAX_WORDS:
        return None
    return passage


def _collect(recipe: Recipe, taken: set[str]) -> list[dict]:
    rng = random.Random(f"{SEED}-{recipe.genre}-{recipe.subset}")
    prefix = f"h-{recipe.genre}" if recipe.subset == "main" else f"h-{recipe.subset}"
    rows: list[dict] = []
    for doc in recipe.documents(rng):
        if len(rows) == recipe.count:
            break
        if doc.doc_id in taken:
            continue
        passage = _excerpt(doc, rng, recipe.from_start)
        if passage is None:
            continue
        taken.add(doc.doc_id)
        passage_id = f"{prefix}-{len(rows) + 1:03d}"
        (paths.HUMAN_DIR / f"{passage_id}.txt").write_text(passage, encoding="utf-8")
        rows.append({
            "id": passage_id,
            "label": "human",
            "subset": recipe.subset,
            "genre": recipe.genre,
            "n_words": word_count(passage),
            "source": doc.source,
            "license": doc.license,
            "title": doc.title,
            "url": doc.url,
            "date": doc.date,
            "cluster": doc.doc_id,
            "writer": "human",
            "detail": doc.detail,
        })
    log.info("%s/%s: %d passages", recipe.genre, recipe.subset, len(rows))
    return rows


def collect_human() -> None:
    if paths.METADATA.exists():
        raise FileExistsError(
            f"{paths.METADATA} already exists; move it away before collecting again"
        )
    paths.HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    rows = [row for recipe in RECIPES for row in _collect(recipe, taken)]
    write_metadata(rows)
