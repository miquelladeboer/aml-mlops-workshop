import os
import onnxruntime


def setup_function():
    """
    Load the model from the outputs folder
    When running integration tests, make sure to populate the outputs
    folder with the model version under tests
    """
    OUTPUTSFOLDER = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../../outputs/models/"
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
