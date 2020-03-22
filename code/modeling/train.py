import torch.nn as nn
import torch

from torch.autograd import Variable

import logging
import numpy as np
from optparse import OptionParser
import sys
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--learning_rate",
              type=float, default=0.01)
op.add_option("--num_epochs",
              type=int, default=4)
op.add_option("--batch_size",
              type=int,
              default=150)
op.add_option("--hidden_size",
              type=int,
              default=100)


argv = []
sys.argv[1:]
(opts, args) = op.parse_args(argv)

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)

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
num_classes = len(np.unique(data_train.target))
# Categories: graphics, scispace and baseball


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


# define metric
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


# Train the Model
epoch_losses = []
epoch_accuracy = []
for epoch in range(num_epochs):
    total_batch = int(len(data_train.data)/batch_size)
    epoch_loss = 0
    epoch_acc = 0
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
        acc = binary_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        # loss and accuracy
        epoch_loss = loss.item()
        epoch_acc = acc

    epoch_losses.append(epoch_loss / total_batch)
    epoch_accuracy.append(epoch_acc)

plt.plot(np.array(epoch_losses), 'r', label="Loss")
plt.xticks(np.arange(1, (num_epochs+1), step=1))
plt.xlabel("Epochs")
plt.ylabel("Loss Percentage")
plt.legend(loc='upper left')
plt.show()

plt.plot(np.array(epoch_accuracy), 'b', label="Accuracy")
plt.xticks(np.arange(1, (num_epochs+1), step=1))
plt.xlabel("Epochs")
plt.ylabel("Accuracy Percentage")
plt.legend(loc='upper left')
plt.show()

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
