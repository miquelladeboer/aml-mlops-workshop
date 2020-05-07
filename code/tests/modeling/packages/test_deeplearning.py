from modeling.packages.deeplearning import get_word_2_index


def test_index_words():
    """ @TODO """
    assert True


def test_get_word_2_index():
    """ Should return index for each unique word """
    vocab = [
        "text",
        "text",
        "water",
        "house",
        "house",
        "water",
        "house"
    ]

    w2i = get_word_2_index(vocab)

    assert w2i.get("house") == 6
    assert w2i.get("water") == 5
    assert w2i.get("text") == 1
    assert w2i.get("unexisting") is None
