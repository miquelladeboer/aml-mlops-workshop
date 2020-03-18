import torch.nn as nn
import torch

from torch.autograd import Variable
import os

import logging
import numpy as np
from optparse import OptionParser
import sys
from collections import Counter

from sklearn.externals import joblib

from azureml.core import Run

OUTPUTSFOLDER = "outputs"

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--learning_rate",
              type=float, default=0.01)
op.add_option("--num_epochs",
              type=int, default=2)
op.add_option("--batch_size",
              type=int,
              default=150)
op.add_option("--hidden_size",
              type=int,
              default=100)


argv = []
sys.argv[1:]
(opts, args) = op.parse_args(argv)

# Retrieve the run and its context (datasets etc.)
run = Run.get_context()

# Load the input datasets from Azure ML
dataset_train = run.input_datasets['subset_train'].to_pandas_dataframe()
dataset_test = run.input_datasets['subset_test'].to_pandas_dataframe()


# Pre-process df for sklearn
class data_train(object):
    def __init__(self, data, target):
        self.data = []
        self.target = []


class data_test(object):
    def __init__(self, data, target):
        self.data = []
        self.target = []


# convert to numpy df
data_train.data = dataset_train.text.values
data_test.data = dataset_test.text.values

# convert label to int
data_train.target = [int(value or 0) for value in dataset_train.target.values]
data_test.target = [int(value or 0) for value in dataset_test.target.values]

########################

vocab = Counter()

for text in data_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

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


# Select the training hyperparameters.
# Create a dict of hyperparameters from the input flags.
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

input_size = total_words  # Words in vocab
num_classes = num_classes = len(np.unique(data_train.target))
# Categories: graphics, sci.space and baseball


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


# input [batch_size, n_labels]
# output [max index for each item in batch, ... ,batch_size-1]
loss = nn.CrossEntropyLoss()
input = Variable(torch.randn(2, 5), requires_grad=True)

target = Variable(torch.LongTensor(2).random_(5))

output = loss(input, target)
output.backward()

net = OurNet(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    total_batch = int(len(data_train.data)/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = get_batch(data_train.data,
                                     data_train.target,
                                     i,
                                     batch_size)
        articles = Variable(torch.FloatTensor(batch_x))
        labels = Variable(torch.LongTensor(batch_y))

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(articles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Test the Model
correct = 0
total = 0
total_test_data = len(data_test.target)
batch_x_test, batch_y_test = get_batch(data_test.data,
                                       data_test.target,
                                       0,
                                       total_test_data)
articles = Variable(torch.FloatTensor(batch_x_test))
labels = torch.LongTensor(batch_y_test)
outputs = net(articles)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum()

correct2 = float(correct)

accuracy = (correct2 / total)

# log metrics
run_logger = Run.get_context()
run_logger.log("accuracy", float(accuracy))

# create outputs folder if not exists
if not os.path.exists(OUTPUTSFOLDER):
    os.makedirs(OUTPUTSFOLDER)

# save .pkl file
model_name = "model" + ".pkl"
filename = os.path.join(OUTPUTSFOLDER, model_name)
joblib.dump(value=outputs, filename=filename)
run_logger.upload_file(name=model_name, path_or_stream=filename)
