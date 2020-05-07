
from collections import Counter
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import json
import os
import pandas as pd


def index_words(data_train, data_test):
    vocab = Counter()

    for text in data_train.text:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    for text in data_test.text:
        for word in text.split(' '):
            vocab[word.lower()] += 1
  
    return vocab, len(vocab)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


def get_batch(df_data, df_target, i, batch_size, total_words, word2index):
    batches = []
    texts = df_data[i*batch_size:i*batch_size+batch_size]
    categories = df_target[i*batch_size:i*batch_size+batch_size]

    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)
    return np.array(batches), np.array(categories)


def get_hyperparameters(opts):
    if opts.fullmodel is True and opts.data_local is False:
        with open(os.environ.get("AZUREML_DATAREFERENCE_metrics_data")) as f:
            metrics_output_result = f.read()
            deserialized_metrics_output = json.loads(metrics_output_result)
            df = pd.DataFrame(deserialized_metrics_output)
            df = df.T

            for column in df:
                df[column] = df[column].astype(str).str.strip("[]").astype(float)

            runID = df.accuracy.idxmax()
            parameters = df.loc[runID].iloc[1:]

            learning_rate = parameters.learning_rate
            num_epochs = int(parameters.num_epochs)
            batch_size = int(parameters.batch_size)
            hidden_size = int(parameters.hidden_size)

    else:
        hyperparameters = {
         "learning_rate": opts.learning_rate,
         "num_epochs": opts.num_epochs,
         "batch_size": opts.batch_size,
         "hidden_size": opts.hidden_size,
        }

        # Select the training hyperparameters.
        learning_rate = hyperparameters["learning_rate"]
        num_epochs = hyperparameters["num_epochs"]
        batch_size = hyperparameters["batch_size"]
        hidden_size = hyperparameters["hidden_size"]
      
    return learning_rate, num_epochs, batch_size, hidden_size


class OurNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(OurNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


def binary_accuracy(preds, y):
    correct = 0
    total = 0
    # round predictions to the closest integer
    _, predicted = torch.max(preds, 1)
    correct += (predicted == y).sum()
    correct2 = float(correct)
    total += y.size(0)
    acc = (correct2 / total)
    return acc


def train_model(num_epochs, data_train, data_test, batch_size,
                criterion, optimizer, net, total_words, word2index):
    epoch_losses = []
    epoch_accuracy = []
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        total_batch = int(len(data_train.text)/batch_size)
        epoch_loss = 0
        epoch_acc = 0
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(data_train.text,
                                         data_train.target,
                                         i,
                                         batch_size,
                                         total_words,
                                         word2index)
            articles = Variable(torch.FloatTensor(batch_x))
            labels = Variable(torch.LongTensor(batch_y))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(articles)
            loss = criterion(outputs, labels)
            acc = binary_accuracy(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss and accuracy
            epoch_loss = loss.item()
            epoch_acc = acc

        epoch_losses.append(epoch_loss / total_batch)
        epoch_accuracy.append(epoch_acc)
        print("epoch accuracy:", epoch_acc)

    return epoch_losses, epoch_accuracy, net


def test_model(data_test, net, total_words, word2index):
    # Test the Model
    correct = 0
    total = 0
    total_test_data = len(data_test.target)
    batch_x_test, batch_y_test = get_batch(data_test.text,
                                           data_test.target,
                                           0,
                                           total_test_data,
                                           total_words,
                                           word2index)
    articles = Variable(torch.FloatTensor(batch_x_test))
    labels = torch.LongTensor(batch_y_test)
    outputs = net(articles)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    correct2 = float(correct)

    accuracy = (correct2 / total)
    return accuracy
