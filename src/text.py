import re
from html.parser import HTMLParser

# Curly quotes, no-break/thin spaces, zero-width space, BOM.
_ASCII_PUNCT = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x00A0: " ", 0x2009: " ", 0x200B: None, 0xFEFF: None,
}
_SENTENCE_BREAK = re.compile(r"""(?<=[.!?])\s+|(?<=[.!?]["')\]])\s+""")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "gen", "gov", "sen", "rep",
    "col", "lt", "capt", "prof", "rev", "hon", "vs", "no", "inc", "u.s", "e.g", "i.e",
}


def normalize(text: str) -> str:
    """Straight quotes, single spaces, paragraphs separated by one blank line.

    Applied to human and AI text alike so that curly quotes or odd whitespace
    can't act as a label.
    """
    text = text.translate(_ASCII_PUNCT)
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p)


def word_count(text: str) -> int:
    return len(text.split())


def _ends_with_abbreviation(piece: str) -> bool:
    last = piece.split()[-1].rstrip(".").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_BREAK.split(paragraph.strip()):
        if not piece:
            continue
        glue = sentences and (
            _ends_with_abbreviation(sentences[-1]) or piece[0].islower()
        )
        if glue:
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return sentences


def cut_passage(paragraphs: list[str], target_words: int, start: int = 0) -> str:
    """Whole sentences from paragraphs[start:], stopping once target_words is reached.

    Returns fewer words than asked if the document runs out.
    """
    taken: list[str] = []
    total = 0
    for paragraph in paragraphs[start:]:
        kept: list[str] = []
        for sentence in split_sentences(paragraph):
            kept.append(sentence)
            total += word_count(sentence)
            if total >= target_words:
                break
        taken.append(" ".join(kept))
        if total >= target_words:
            break
    return "\n\n".join(taken)


class _ParagraphParser(HTMLParser):
    # Footnote markers, tables, code and math are not prose and are dropped.
    SKIPPED_TAGS = {"sup", "style", "script", "table", "math", "pre", "code", "figure"}

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._open: list[str] | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag == "p" and not self._skip_depth:
            self._open = []

    def handle_endtag(self, tag):
        if tag in self.SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "p" and self._open is not None:
            self.paragraphs.append(" ".join("".join(self._open).split()))
            self._open = None

    def handle_data(self, data):
        if self._open is not None and not self._skip_depth:
            self._open.append(data)


def html_paragraphs(html: str) -> list[str]:
    parser = _ParagraphParser()
    parser.feed(html)
    return [p for p in parser.paragraphs if p]
