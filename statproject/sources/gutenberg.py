import csv
import random
import re
import time
from collections.abc import Iterator

import requests

from ..paths import CACHE
from . import USER_AGENT, Document

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
# gutenberg.org asks scripts to use a mirror rather than the main site.
MIRROR = "https://aleph.pglaf.org/cache/epub/{id}/pg{id}.txt"
REQUEST_PAUSE = 1.0
FICTION_CLASSES = {"PR", "PS", "PZ"}
# Authors born after this write prose that is old but not archaic.
EARLIEST_BIRTH_YEAR = 1840
_BIRTH_YEAR = re.compile(r", (\d{4})-")
_BODY = re.compile(r"\*\*\* ?START OF.*?\*\*\*(.*?)\*\*\* ?END OF", re.DOTALL)
# Skip front and back matter, which is mostly contents pages and adverts.
BODY_SLICE = (0.15, 0.9)
EM_DASH = chr(0x2014)


def _catalog() -> list[dict]:
    path = CACHE / "pg_catalog.csv"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        response = requests.get(CATALOG_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_candidate(row: dict) -> bool:
    born = _BIRTH_YEAR.search(row["Authors"])
    return (
        row["Type"] == "Text"
        and row["Language"] == "en"
        and "fiction" in row["Subjects"].lower()
        and bool(FICTION_CLASSES & set(row["LoCC"].split("; ")))
        and born is not None
        and int(born.group(1)) >= EARLIEST_BIRTH_YEAR
    )


def _body_paragraphs(raw: str) -> list[str]:
    match = _BODY.search(raw)
    if match is None:
        return []
    # Plain-text conventions: underscores for italics, double hyphens for dashes.
    body = match.group(1).replace("_", "").replace("--", EM_DASH)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\r?\n\s*\r?\n", body)]
    paragraphs = [p for p in paragraphs if p and not p.startswith("[")]
    lo, hi = (int(len(paragraphs) * frac) for frac in BODY_SLICE)
    return paragraphs[lo:hi]


def fiction(rng: random.Random) -> Iterator[Document]:
    """Public-domain novels and stories, at most one book per author."""
    books = [row for row in _catalog() if _is_candidate(row)]
    rng.shuffle(books)
    seen_authors: set[str] = set()
    for book in books:
        if book["Authors"] in seen_authors:
            continue
        seen_authors.add(book["Authors"])
        book_id = book["Text#"]
        response = requests.get(MIRROR.format(id=book_id), headers={"User-Agent": USER_AGENT}, timeout=60)
        time.sleep(REQUEST_PAUSE)
        if not response.ok:
            continue
        response.encoding = "utf-8"
        yield Document(
            doc_id=f"gutenberg:{book_id}",
            title=f"{book['Title']} ({book['Authors']})",
            url=f"https://www.gutenberg.org/ebooks/{book_id}",
            date="before 1964",
            source="Project Gutenberg",
            license="Public domain (US)",
            paragraphs=_body_paragraphs(response.text),
        )
