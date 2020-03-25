import json
import numpy as np
import onnxruntime
import os
import time
import torch
from torch.autograd import Variable
from azureml.core.conda_dependencies import CondaDependencies


def init():
    global session, input_name, output_name
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'net.onnx')
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
  

def preprocess(input_data_json):

    input_data = json.loads(input_data_json)

    data_test = {}
    data_test["data"] = input_data.text
    data_test["target"] = input_data.target

    vocab = []

    for text in data_test.data:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    total_words = len(vocab)

    def get_word_2_index(vocab):
        word2index = {}
        for i, word in enumerate(vocab):
            word2index[word.lower()] = i

        return word2index

    word2index = get_word_2_index(vocab)

    def get_batch(df_data, df_target, i, batch_size):
        batches = []
        texts = df_data[i*batch_size:i*batch_size+batch_size]
        categories = df_target[i*batch_size:i*batch_size+batch_size]

        for text in texts:
            layer = np.zeros(total_words, dtype=float)
            for word in text.split(' '):
                layer[word2index[word.lower()]] += 1
            batches.append(layer)
        return np.array(batches), np.array(categories)

    total_test_data = len(data_test.target)
    batch_x_test, batch_y_test = get_batch(data_test.data,
                                           data_test.target,
                                           0,
                                           total_test_data)
    articles = Variable(torch.FloatTensor(batch_x_test))
    labels = torch.LongTensor(batch_y_test)

    return articles, labels


def postprocess(result):
    _, predicted = torch.max(result.data, 1)
    return predicted


def run(input_data):

    try:
        # load in our data, convert to readable format
        articles, labels = preprocess(input_data)

        # start timer
        start = time.time()

        r = session.run([output_name], {input_name: data})
       
        #end timer
        end = time.time()
       
        result = postprocess(r)
        result_dict = {"result": result,
                      "time_in_sec": end - start}
    except Exception as e:
        result_dict = {"error": str(e)}
    
    return result_dict

#def choose_class(result_prob):
#    """We use argmax to deterftrmine the right label to choose from our output"""
#    return int(np.argmax(result_prob, axis=0))


if __name__ == "__main__":
   sample_input = {
       "text": "it's late on the evening"
   }

   result = run(sample_input)

   print(result)

