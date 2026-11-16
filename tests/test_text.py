from statproject.text import cut_passage, html_paragraphs, normalize, split_sentences


def test_split_sentences_keeps_titles_and_initials_together():
    text = 'Mr. Murphy spoke first. "We respect that," he said. J. K. Rowling agreed!'
    assert split_sentences(text) == [
        "Mr. Murphy spoke first.",
        '"We respect that," he said.',
        "J. K. Rowling agreed!",
    ]


def test_cut_passage_stops_at_a_sentence_end():
    paragraphs = ["One two three. Four five six seven.", "Eight nine. Ten eleven twelve."]
    assert cut_passage(paragraphs, target_words=5) == "One two three. Four five six seven."
    assert cut_passage(paragraphs, target_words=8) == "One two three. Four five six seven.\n\nEight nine."
    assert cut_passage(paragraphs, target_words=3, start=1) == "Eight nine. Ten eleven twelve."


def test_cut_passage_returns_what_exists_when_the_document_is_short():
    assert cut_passage(["Only this."], target_words=50) == "Only this."


def test_normalize_straightens_quotes_and_whitespace():
    curly = chr(0x201C) + "Hi" + chr(0x201D) + " it" + chr(0x2019) + "s\n  me.\n\n\n\nNext" + chr(0xA0) + "one."
    assert normalize(curly) == '"Hi" it\'s me.\n\nNext one.'


def test_html_paragraphs_drops_footnotes_and_tables():
    html = "<p>Kept<sup>[1]</sup> text.</p><table><tr><td><p>cell</p></td></tr></table><p>Second <b>one</b>.</p>"
    assert html_paragraphs(html) == ["Kept text.", "Second one."]
