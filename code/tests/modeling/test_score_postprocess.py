import pytest
from modeling.score import postprocess


def test_postprocess_datasamplea():
    mock_result = [0.8, 0.6, 0.4, 0.9]

    x = postprocess(mock_result)
    assert x == 'sci.space'


def test_postprocess_datasampleb():
    mock_result = [0.8, 0.6, 0.85, 0.3]

    x = postprocess(mock_result)
    assert x == 'comp.graphics'


def test_postprocess_datasamplec():
    mock_result = [0.9, 0.6, 0.85, 0.3]

    x = postprocess(mock_result)
    assert x == 'alt.atheism'


def test_postprocess_unknownresultindex():
    mock_result = [0.9, 0.6, 0.85, 0.3, 0.95]

    with pytest.raises(ValueError):
        x = postprocess(mock_result)
        print(x)

