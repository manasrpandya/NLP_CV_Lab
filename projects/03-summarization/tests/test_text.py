from summ.extractive import lead, textrank
from summ.text import sents, words

toy = (
    "Mr. Smith went to Washington on Monday. He met Dr. Jones there. They talked about the budget for hours. "
    "The budget was large. Nobody in the room liked the budget. Then it rained all night."
)


def test_sents_keeps_abbreviations():
    s = sents(toy)
    assert s[0] == "Mr. Smith went to Washington on Monday."
    assert s[1] == "He met Dr. Jones there."
    assert len(s) == 6


def test_words():
    assert words("It's 5 p.m., Don't go!") == ["it's", "5", "p", "m", "don't", "go"]


def test_lead():
    assert lead(toy, 2) == "Mr. Smith went to Washington on Monday. He met Dr. Jones there."


def test_textrank_prefers_central_sentences_in_order():
    s = sents(toy)
    out = textrank(toy, 2)
    got = sents(out)
    assert len(got) == 2
    # the two budget sentences share the most vocabulary with the rest
    assert any("budget" in x for x in got)
    # kept in source order
    assert [s.index(x) for x in got] == sorted(s.index(x) for x in got)


def test_textrank_short_text_returned_whole():
    assert textrank("One sentence only here.", 3) == "One sentence only here."
