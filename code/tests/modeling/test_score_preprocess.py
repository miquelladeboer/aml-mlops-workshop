import os
from modeling.score import preprocess


def setup_function():
    """
    The pre-processing function relies on a word2index
    vocubalary file, which is changing with each model iteration.
    To test accordingly, we mock the word2index file (to be static)
    and pass it along with the tests and in the test directory.
    """
    OUTPUTSFOLDER = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "mock_outputs"
    ),
    WORD2INDEX_PKL = "word2index"

    # @TODO create mock word2index


def test_preprocess_datasamplea():
    """@TODO"""
    return True


def test_preprocess_datasampleb():
    """@TODO"""
    return True


def test_preprocess_datasamplec():
    """@TODO"""
    return True


def test_preprocess_invalidformat():
    """@TODO"""
    return True
