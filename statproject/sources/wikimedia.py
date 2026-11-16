"""Wikipedia, Wikinews and Wikisource, all through the MediaWiki API."""

import datetime as dt
import random
import re
import time
from collections.abc import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..text import html_paragraphs
from . import CUTOFF_DATE, USER_AGENT, Document

# Clients without contact details in their User-Agent get 10 requests a minute.
REQUEST_PAUSE = 6.5
VITAL_ARTICLES_PAGE = "Wikipedia:Vital articles"
WIKINEWS_FIRST_DAY = dt.date(2005, 1, 1)
SOTU_CATEGORY = "Category:State of the Union addresses"
# Earlier addresses are written messages in a very different idiom.
POSTWAR_PRESIDENTS = (
    "Truman", "Eisenhower", "Kennedy", "Lyndon", "Nixon", "Ford", "Carter",
    "Reagan", "Bush", "Clinton", "Obama", "Trump",
)
_WEEKDAY_DATELINE = re.compile(r"^(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day, \w+ \d{1,2}, \d{4}")
_SKIPPED_NEWS_TITLES = re.compile(r"interview|shorts|briefs|digest|wikinews", re.IGNORECASE)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
# Wikimedia answers bursts with 429 and a Retry-After header, which Retry honours.
_session.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=2, status_forcelist=(429, 503))))


def _api(site: str, **params) -> dict:
    params |= {"format": "json", "formatversion": 2}
    response = _session.get(f"https://{site}/w/api.php", params=params, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_PAUSE)
    return response.json()


def _revision_before_cutoff(site: str, title: str) -> tuple[int, str] | None:
    data = _api(
        site, action="query", prop="revisions", titles=title, redirects=1,
        rvlimit=1, rvprop="ids|timestamp", rvstart=f"{CUTOFF_DATE}T23:59:59Z",
    )
    revisions = data["query"]["pages"][0].get("revisions")
    if not revisions:
        return None
    return revisions[0]["revid"], revisions[0]["timestamp"][:10]


def _revision_paragraphs(site: str, revid: int) -> list[str]:
    data = _api(site, action="parse", oldid=revid, prop="text")
    # Hidden or deleted revisions come back as an error object instead.
    if "parse" not in data:
        return []
    return html_paragraphs(data["parse"]["text"])


def _category_members(site: str, category: str) -> list[str]:
    data = _api(
        site, action="query", list="categorymembers", cmtitle=category,
        cmnamespace=0, cmlimit="max",
    )
    return [member["title"] for member in data["query"]["categorymembers"]]


def _page_links(site: str, page: str) -> list[str]:
    titles: list[str] = []
    params = dict(action="query", prop="links", titles=page, plnamespace=0, pllimit="max")
    while True:
        data = _api(site, **params)
        titles += [link["title"] for link in data["query"]["pages"][0].get("links", [])]
        if "continue" not in data:
            return titles
        params |= data["continue"]


def _document(site: str, title: str, source: str, license: str) -> Document | None:
    revision = _revision_before_cutoff(site, title)
    if revision is None:
        return None
    revid, date = revision
    return Document(
        doc_id=f"{site.split('.')[1]}:{revid}",
        title=title,
        url=f"https://{site}/w/index.php?oldid={revid}",
        date=date,
        source=source,
        license=license,
        paragraphs=_revision_paragraphs(site, revid),
    )


def wikipedia(rng: random.Random) -> Iterator[Document]:
    """Level-3 vital articles as they stood at the cutoff date."""
    site = "en.wikipedia.org"
    titles = sorted(set(_page_links(site, VITAL_ARTICLES_PAGE)))
    rng.shuffle(titles)
    for title in titles:
        doc = _document(site, title, "Wikipedia", "CC BY-SA 3.0")
        if doc:
            yield doc


def wikinews(rng: random.Random) -> Iterator[Document]:
    """One article from each of a random sequence of days, 2005 through the cutoff."""
    site = "en.wikinews.org"
    last_day = dt.date.fromisoformat(CUTOFF_DATE)
    span = (last_day - WIKINEWS_FIRST_DAY).days
    while True:
        day = WIKINEWS_FIRST_DAY + dt.timedelta(days=rng.randrange(span + 1))
        titles = _category_members(site, f"Category:{day:%B} {day.day}, {day.year}")
        titles = [t for t in titles if not _SKIPPED_NEWS_TITLES.search(t)]
        if not titles:
            continue
        doc = _document(site, rng.choice(titles), "Wikinews", "CC BY 2.5")
        if doc is None:
            continue
        doc.date = day.isoformat()
        doc.paragraphs = [p for p in doc.paragraphs if not _WEEKDAY_DATELINE.match(p)]
        yield doc


def state_of_the_union(rng: random.Random) -> Iterator[Document]:
    site = "en.wikisource.org"
    titles = [
        title for title in _category_members(site, SOTU_CATEGORY)
        if any(name in title for name in POSTWAR_PRESIDENTS)
    ]
    rng.shuffle(titles)
    for title in titles:
        doc = _document(site, title, "Wikisource (State of the Union addresses)", "Public domain (US government work)")
        if doc:
            yield doc
