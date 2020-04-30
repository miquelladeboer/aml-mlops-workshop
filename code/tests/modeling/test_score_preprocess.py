# The pre-processing function relies on a word2index
# vocubalary file, that is changing with each model iteration.
# To test accordingly, we mock the word2index file (to be static) 
# and pass it along with the tests and in the test directory.
#
# @TODO create mock word2index 
# @TODO refactor reference to specific word2index file from preprocess function
# @TODO pass mock word2index with tests
from modeling.score import preprocess


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
