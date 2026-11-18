import datetime as dt
import html
import random
import time
from collections.abc import Iterator

import requests

from ..text import html_paragraphs
from . import CUTOFF_DATE, Document

API = "https://api.stackexchange.com/2.3"
SITES = (
    "cooking", "travel", "diy", "workplace", "parenting",
    "movies", "gardening", "outdoors", "money", "academia",
)
# Moderately upvoted answers: real people liked them, but they aren't the
# heavily edited showpieces at the top of each site.
MIN_SCORE, MAX_SCORE = 5, 40
PAGES_TO_DRAW_FROM = 20
NON_PROSE_MARKUP = ("<pre", "<code", "<blockquote", "<img", "<li", "<table")
_CUTOFF_EPOCH = int(dt.datetime.fromisoformat(f"{CUTOFF_DATE}T23:59:59+00:00").timestamp())


def _get(path: str, **params) -> list[dict]:
    response = requests.get(f"{API}/{path}", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    # The API asks clients to wait when it sends a backoff value.
    time.sleep(payload.get("backoff", 0) + 0.5)
    return payload["items"]


def _untouched_prose(answer: dict) -> bool:
    edited_after_cutoff = answer.get("last_edit_date", 0) > _CUTOFF_EPOCH
    return not edited_after_cutoff and not any(m in answer["body"] for m in NON_PROSE_MARKUP)


def _site_answers(site: str, rng: random.Random) -> list[Document]:
    answers = _get(
        "answers", site=site, sort="votes", order="desc", min=MIN_SCORE, max=MAX_SCORE,
        todate=_CUTOFF_EPOCH, pagesize=100, page=rng.randint(1, PAGES_TO_DRAW_FROM),
        filter="withbody",
    )
    answers = [a for a in answers if _untouched_prose(a)]
    rng.shuffle(answers)
    question_ids = ";".join(str(a["question_id"]) for a in answers)
    questions = _get(f"questions/{question_ids}", site=site, pagesize=100)
    titles = {q["question_id"]: q["title"] for q in questions}
    return [
        Document(
            doc_id=f"stackexchange:{site}:{a['answer_id']}",
            title=html.unescape(titles[a["question_id"]]),
            url=f"https://{site}.stackexchange.com/a/{a['answer_id']}",
            date=dt.datetime.fromtimestamp(a["creation_date"], dt.UTC).date().isoformat(),
            source=f"Stack Exchange ({site})",
            license=a["content_license"],
            paragraphs=html_paragraphs(a["body"]),
        )
        for a in answers
        if a["question_id"] in titles
    ]


def answers(rng: random.Random) -> Iterator[Document]:
    """Answers from a spread of non-programming sites, dealt round-robin."""
    per_site = [_site_answers(site, rng) for site in SITES]
    for round_ in zip(*per_site):
        yield from round_
