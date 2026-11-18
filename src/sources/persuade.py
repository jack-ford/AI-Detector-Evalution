import csv
import random
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from functools import cache

import requests

from ..paths import CACHE
from . import Document

DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=13phHyDzIsb0MHyJr6q-B-qIa9P2tM135&export=download&confirm=t"
)
REPO_URL = "https://github.com/scrosseye/persuade_corpus_2.0"
# Anonymisation placeholders like STUDENT_NAME or Generic_City would be a giveaway.
_PLACEHOLDER = re.compile(r"\b[A-Z]+_[A-Z_]+\b|Generic_\w+")


@cache
def _essays() -> list[dict]:
    path = CACHE / "persuade_train.csv"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        with requests.get(DOWNLOAD_URL, stream=True, timeout=600) as response:
            response.raise_for_status()
            with path.open("wb") as f:
                for block in response.iter_content(chunk_size=1 << 20):
                    f.write(block)
    # The file has one row per annotated discourse element, each repeating the
    # whole essay, so fields are far longer than the csv module's default limit.
    csv.field_size_limit(sys.maxsize)
    essays: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            essays.setdefault(row["essay_id"], row)
    return [e for e in essays.values() if not _PLACEHOLDER.search(e["full_text"])]


def _document(essay: dict) -> Document:
    return Document(
        doc_id=f"persuade:{essay['essay_id']}",
        title=essay["prompt_name"],
        url=REPO_URL,
        date="2010-2020",
        source="PERSUADE 2.0",
        license="CC BY 4.0",
        paragraphs=[p for p in essay["full_text"].splitlines() if p.strip()],
        detail=(
            f"grade={essay['grade_level']};ell={essay['ell_status'].strip()};"
            f"holistic_score={essay['holistic_essay_score']}"
        ),
    )


def _by_prompt(essays: list[dict], rng: random.Random) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for essay in essays:
        groups[essay["prompt_name"]].append(essay)
    for group in groups.values():
        rng.shuffle(group)
    return [groups[prompt] for prompt in sorted(groups)]


def essays(rng: random.Random) -> Iterator[Document]:
    for round_ in zip(*_by_prompt(_essays(), rng)):
        for essay in round_:
            yield _document(essay)


def ell_matched_pairs(rng: random.Random) -> Iterator[Document]:
    pool = _essays()
    learners = [e for e in pool if e["ell_status"].strip() == "Yes"]
    others: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for essay in pool:
        if essay["ell_status"].strip() == "No":
            others[essay["prompt_name"], essay["grade_level"]].append(essay)
    for group in others.values():
        rng.shuffle(group)
    for round_ in zip(*_by_prompt(learners, rng)):
        for learner in round_:
            matches = others[learner["prompt_name"], learner["grade_level"]]
            if matches:
                yield _document(learner)
                yield _document(matches.pop())
