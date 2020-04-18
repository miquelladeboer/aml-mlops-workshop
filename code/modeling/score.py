import json
import numpy as np
import onnxruntime
import os
import time
import torch
from torch.autograd import Variable
import pickle
import string


def init():
    global session, input_name, output_name
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path ./azureml-models/$MODEL_NAME/$VERSION
    model = os.path.join(os.environ['AZUREML_MODEL_DIR'], 'net.onnx')
    session = onnxruntime.InferenceSession(model, None)
    print(session)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess(input_data_json):

    input_data = json.loads(input_data_json)

    # print("Parsed json {}".format(input_data))
    OUTPUTSFOLDER = "outputs"
    pickle_name = "word2index"
    filename = os.path.join(OUTPUTSFOLDER, pickle_name)
    infile = open(filename, 'rb')
    word2index = pickle.load(infile)
    infile.close()

    data_new = {}
    data_new["data"] = input_data['text']

    def get_batch(df_data):
        batches = []
        texts = df_data.split(' ')
        layer = np.zeros(len(word2index), dtype=float)
        table = str.maketrans('', '', string.punctuation)
        for text in texts:
            word = text.translate(table)
            try:
                layer[word2index[word.lower()]] += 1
            except (KeyError):
                pass
        batches.append(layer)
        return np.array(batches)

    batch_x_test = get_batch(data_new["data"])
    print(batch_x_test)

    return Variable(torch.FloatTensor(batch_x_test))


def postprocess(result):
    print(result)
    index = np.argmax(result)
    if index == 0:
        predicted = 'alt.atheism'
    elif index == 1:
        predicted = 'talk.religion.misc'
    elif index == 2:
        predicted = 'comp.graphics'
    else:
        predicted = 'sci.space'
    return predicted


def run(input_data):

    try:
        # load in our data, convert to readable format
        preprocessed_input = preprocess(input_data)

        # start timer
        start = time.time()

        # print("pre-processed input {}".format(preprocessed_input))

        # predict
        r = session.run(
            [output_name],
            {
                input_name: to_numpy(preprocessed_input)
            }
        )

        # end timer
        end = time.time()

        result = postprocess(r)

        result_dict = {
            "result": result,
            "time_in_sec": end - start
        }

    except Exception as e:
        result_dict = {"error": str(e)}

    return result_dict

# def choose_class(result_prob):
#    """We use argmax to deterftrmine the right label to choose from our
#  output"""
#    return int(np.argmax(result_prob, axis=0))


if __name__ == "__main__":
    # set local env var for testing purposes
    os.environ["AZUREML_MODEL_DIR"] = "outputs"

    # load model
    init()

    # create sample response
    sample_input = json.dumps({
        "text": "Subject: Re: Biblical Backing of 's 3-02 Tape (Cites enclosed) From: kmcvay@oneb.almanac.bc.ca (Ken Mcvay) Organization: The Old Frog's Almanac Lines: 20 In article <20APR199301460499@utarlg.uta.edu> b645zaw@utarlg.uta.edu (stephen) writes: >Seems to me Koresh is yet another messenger that got killed >for the message he carried. (Which says nothing about the Seems to be, barring evidence to the contrary, that Koresh was simply another deranged fanatic who thought it neccessary to take a whole bunch of folks with him, children and all, to satisfy his delusional mania. Jim Jones, circa 1993. >In the mean time, we sure learned a lot about evil and corruption. >Are you surprised things have gotten that rotten? Nope - fruitcakes like Koresh have been demonstrating such evil corruption for centuries. -- The Old Frog's Almanac - A Salute to That Old Frog Hisse'f, Ryugen Fisher (604) 245-3205 (v32) (604) 245-4366 (2400x4) SCO XENIX 2.3.2 GT Ladysmith, British Columbia, CANADA. Serving Central Vancouver Island"
    })

    print("json string {}".format(sample_input))

    result = run(sample_input)

    print(result)
