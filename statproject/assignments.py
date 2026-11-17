"""Plan the AI half of the corpus and register it once it has been written.

This module decides what each AI passage should be (genre, style, length,
which human passage it shadows) and records the instruction. It writes no
passage text: every file in data/ai was written directly by the model, one
passage at a time, following the instruction stored here.
"""

import logging
import random

import pandas as pd

from . import paths
from .corpus import METADATA_COLUMNS, load_metadata, passage_path
from .text import normalize, word_count

log = logging.getLogger(__name__)

SEED = 2026
N_WRITERS = 6
# Share of a genre's human passages that get an AI counterpart in each style
# (18, 8, 8, 8 and 8 for a genre of 50). The rest stay unpaired.
STYLE_SHARES = {"plain": 0.36, "student": 0.16, "anti_ai": 0.16, "casual": 0.16}
PARAPHRASE_SHARE = 0.16
AI_SOURCE = "Claude Fable 5.2, written directly for this project"

GENRE_BRIEFS = {
    "forum": "a forum answer to the question",
    "fiction": "a work of prose fiction comparable to",
    "student": "a school essay on the assigned topic",
}
STYLE_INSTRUCTIONS = {
    "plain": "No style instructions: write it the way you naturally would.",
    "student": (
        "Write it the way an ordinary American high school student would: simple "
        "vocabulary, some filler and awkward phrasing, a few small grammar or "
        "punctuation slips, not polished."
    ),
    "anti_ai": (
        "Make sure it does not read as AI-written: no stock phrases, no tidy "
        "three-part lists, no hedging boilerplate, no neat summary sentence at "
        "the end. Vary sentence length a lot and prefer specific, concrete "
        "details and plain words."
    ),
    "casual": (
        "Write it fast and sloppy, like someone typing on a phone or dashing off "
        "a post: informal, contractions, run-ons or fragments, a few typos, "
        "careless capitalisation and punctuation."
    ),
}
PASSAGE_PROMPT = (
    "Write a roughly {n_words}-word excerpt from {brief} \"{title}\". Human passage "
    "{pair_id} is an excerpt of the same kind: match its subject, its length and "
    "where in the document it sits, but do not reuse its wording. {style}"
)
IMITATION_PROMPT = (
    "Write a roughly {n_words}-word excerpt from a school essay on the assigned topic "
    "\"{title}\", as {author} would write it. Human passage {pair_id} is a "
    "real student's essay on the same assignment: match its length, its level of polish "
    "and its command of English, mistakes included, without reusing its sentences."
)
PARAPHRASE_PROMPT = (
    "Paraphrase AI passage {source}: keep the content, the order of ideas and the "
    "length, but change the wording and the sentence structure throughout."
)


def _genre_plan(genre: str, humans: pd.DataFrame, rng: random.Random) -> list[dict]:
    pairs = humans.sample(frac=1, random_state=rng.randrange(2**32)).to_dict("records")
    styles = [
        style for style, share in STYLE_SHARES.items() for _ in range(round(share * len(pairs)))
    ]
    rows = []
    for k, (style, human) in enumerate(zip(styles, pairs), start=1):
        rows.append({
            "id": f"a-{genre}-{k:03d}",
            "subset": "main",
            "genre": genre,
            "prompt_style": style,
            "pair_id": human["id"],
            "target_words": human["n_words"],
            "paraphrase_of": "",
            # Dealt in order, so every writer gets the same mix of styles.
            "writer": f"w{(k - 1) % N_WRITERS + 1}",
            "prompt": PASSAGE_PROMPT.format(
                n_words=human["n_words"], brief=GENRE_BRIEFS[genre], title=human["title"],
                pair_id=human["id"], style=STYLE_INSTRUCTIONS[style],
            ),
        })
    plain = [row for row in rows if row["prompt_style"] == "plain"]
    n_paraphrases = round(PARAPHRASE_SHARE * len(pairs))
    for k, source in enumerate(rng.sample(plain, n_paraphrases), start=len(rows) + 1):
        rows.append({
            "id": f"a-{genre}-{k:03d}",
            "subset": "main",
            "genre": genre,
            "prompt_style": "paraphrase",
            "pair_id": source["pair_id"],
            "target_words": source["target_words"],
            "paraphrase_of": source["id"],
            # Paraphrases are written afterwards in the main session, not by
            # the writer of the source passage.
            "writer": "w0",
            "prompt": PARAPHRASE_PROMPT.format(source=source["id"]),
        })
    return rows


def _imitation_plan(essays: pd.DataFrame) -> list[dict]:
    rows = []
    for k, essay in enumerate(essays.to_dict("records"), start=1):
        # A few essays in PERSUADE have no grade recorded.
        grade = essay["detail"].split(";")[0].removeprefix("grade=")
        author = f"a grade {grade} student" if grade else "a secondary school student"
        rows.append({
            "id": f"a-imitation-{k:03d}",
            "subset": "imitation_probe",
            "genre": "student",
            "prompt_style": "imitation",
            "pair_id": essay["id"],
            "target_words": essay["n_words"],
            "paraphrase_of": "",
            "writer": f"w{(k - 1) % N_WRITERS + 1}",
            "prompt": IMITATION_PROMPT.format(
                n_words=essay["n_words"], title=essay["title"], author=author, pair_id=essay["id"],
            ),
        })
    return rows


def plan_ai_passages() -> pd.DataFrame:
    if paths.ASSIGNMENTS.exists():
        raise FileExistsError(f"{paths.ASSIGNMENTS} already exists")
    metadata = load_metadata()
    humans = metadata.query("label == 'human' and subset == 'main'")
    rng = random.Random(SEED)
    rows = [
        row
        for genre, group in humans.groupby("genre", sort=True)
        for row in _genre_plan(genre, group, rng)
    ]
    rows += _imitation_plan(metadata.query("subset == 'imitation_probe'"))
    plan = pd.DataFrame(rows)
    paths.AI_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_csv(paths.ASSIGNMENTS, index=False)
    log.info("planned %d AI passages", len(plan))
    return plan


def register_ai_passages() -> None:
    """Add the written AI passages to the metadata file.

    The only edit made to the text is the same whitespace and quote
    normalisation the human passages went through.
    """
    plan = pd.read_csv(paths.ASSIGNMENTS, dtype=str, keep_default_na=False)
    missing = [i for i in plan["id"] if not passage_path(i).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} AI passages not written yet, e.g. {missing[:5]}")

    rows = []
    for row in plan.to_dict("records"):
        path = passage_path(row["id"])
        text = normalize(path.read_text(encoding="utf-8"))
        path.write_text(text, encoding="utf-8")
        rows.append({
            "id": row["id"],
            "label": "ai",
            "subset": row["subset"],
            "genre": row["genre"],
            "prompt_style": row["prompt_style"],
            "n_words": word_count(text),
            "source": AI_SOURCE,
            "license": "none claimed",
            "title": "",
            "cluster": row["paraphrase_of"] or row["id"],
            "pair_id": row["pair_id"],
            "writer": row["writer"],
        })
    humans = load_metadata().query("label == 'human'")
    combined = pd.concat([humans, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.reindex(columns=METADATA_COLUMNS).fillna("")
    combined.to_csv(paths.METADATA, index=False)
    log.info("metadata now lists %d human and %d AI passages", len(humans), len(rows))
