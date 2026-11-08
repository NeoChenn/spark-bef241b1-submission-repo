from policy.control.schema import LINE_CHARS, MAX_LINES, situation_lines


def test_one_long_paragraph_splits_into_sentences():
    text = "All three robots are working with 89% battery. All gas sensors read 0.0% LEL and weather is calm. R2 is welding."
    assert situation_lines(text) == ["All three robots are working with 89% battery.",
                                     "All gas sensors read 0.0% LEL and weather is calm.", "R2 is welding."]


def test_long_sentence_clipped_at_a_word():
    out = situation_lines("word " * 60)
    assert len(out) == 1 and len(out[0]) <= LINE_CHARS and out[0].endswith("word…")


def test_line_cap():
    assert len(situation_lines("A. B. C. D. E.")) == MAX_LINES
