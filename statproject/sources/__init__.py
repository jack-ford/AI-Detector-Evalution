from dataclasses import dataclass

# Everything collected has to predate widely available LLMs.
CUTOFF_DATE = "2020-12-31"
USER_AGENT = "StatProject-corpus-builder/1.0 (school statistics project; python-requests)"


@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    date: str
    source: str
    license: str
    paragraphs: list[str]
    detail: str = ""
