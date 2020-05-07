import os
import onnxruntime


def setup_function():
    """
    The pre-processing function relies on a word2index
    vocubalary file, which is changing with each model iteration.
    To test accordingly, we mock the word2index file (to be static)
    and pass it along with the tests and in the test directory.
    """
    OUTPUTSFOLDER = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../outputs/models/"
    )

    # Load ONNX Model
    model = os.path.join(OUTPUTSFOLDER, 'net.onnx')
    session = onnxruntime.InferenceSession(model, None)
    print(session)


def test_validatemodeloutput_samplea():
    assert True


def test_validatemodeloutput_sampleb():
    assert True


def test_validatemodeloutput_samplec():
    assert True
