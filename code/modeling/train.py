import logging
import argparse
import numpy as np
from sklearn.externals import joblib
import os

from packages.get_data import (load_data_from_local,
                               load_data_from_azure)
from packages.sklearn import (Model_choice,
                              fit_sklearn,
                              pandas_to_numpy,
                              vectorizer)
from packages.plots import (plot_auc, plot_loss_per_epoch,
                            plot_accuracy_per_epoch)
from packages.deeplearning import (index_words,
                                   get_word_2_index,
                                   get_hyperparameters,
                                   OurNet, train_model,
                                   test_model)

from azureml.core import Run

# Get run context
run = Run.get_context()

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default='subset_')
parser.add_argument("--models",
                    type=str,
                    default='sklearnmodels')
parser.add_argument("--fullmodel",
                    type=bool,
                    default=False)
parser.add_argument("--data_local",
                    type=bool,
                    default=True)

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.1)
parser.add_argument("--num_epochs",
                    type=int,
                    default=1)
parser.add_argument("--batch_size",
                    type=int,
                    default=100)
parser.add_argument("--hidden_size",
                    type=int,
                    default=100)

opts = parser.parse_args()

if opts.models == 'deeplearning':
    import torch.nn as nn
    from torch.autograd import Variable
    import torch
    import pickle
    import onnx

if opts.data_local is True:
    # load data from local path
    data_train, data_test = load_data_from_local(opts.dataset)
else:
    # load data from Azure
    data_train, data_test = load_data_from_azure(opts.dataset, run)

if opts.models != 'deeplearning':
    # get numpy back
    X_train, X_test, y_train, y_test = pandas_to_numpy(data_train, data_test)
    # vectorize data
    X_train, X_test = vectorizer(X_train, X_test)
    # Get models from sklearn
    models = Model_choice(opts.models)

    # Run benchmark and collect results with multiple classifiers
    for i in range(models.number_of_models()):
        clf, name = models.select_model(i)

        (clf_descr, accuracy, balanced_accuracy,
         precision, recall, f1, fpr, tpr, roc_auc) = fit_sklearn(
            clf, X_train, X_test, y_train, y_test)

        # plot auc
        plt = plot_auc(fpr, tpr, roc_auc)

        # child runs
        if opts.models == 'sklearnmodels':
            # create a child run for Azure ML logging
            child_run = run.child_run(name=name)
            child_run.log("accuracy", float(accuracy))
            child_run.log("balanced accuracy", float(balanced_accuracy))
            child_run.log("F1 score", float(f1))
            child_run.log("precision", float(precision))
            child_run.log("recall", float(recall))
            child_run.log_image("AUC"+name, plot=plt)

            child_run.complete()

        # log score to AML
        run.log("accuracy", float(accuracy))
        run.log("balanced accuracy", float(balanced_accuracy))
        run.log("F1 score", float(f1))
        run.log("precision", float(precision))
        run.log("recall", float(recall))
        run.log_image("AUC"+name, plot=plt)

        # write model artifact to AML
        model_name = "model" + str(name) + ".pkl"
        filename = "outputs/models" + model_name
        joblib.dump(value=clf, filename=filename)
        run.upload_file(name=model_name, path_or_stream=filename)

else:
    vocab, total_words = index_words(data_train, data_test)
    word2index = get_word_2_index(vocab)
    (learning_rate, num_epochs,
        batch_size, hidden_size) = get_hyperparameters(opts)
    input_size = total_words  # Words in vocab
    num_classes = len(np.unique(data_train.target))

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

    epoch_losses, epoch_accuracy, net = train_model(num_epochs,
                                                    data_train,
                                                    data_test,
                                                    batch_size,
                                                    criterion,
                                                    optimizer,
                                                    net,
                                                    total_words,
                                                    word2index)

    accuracy = test_model(data_test, net, total_words, word2index)

    plt_loss = plot_loss_per_epoch(epoch_losses, num_epochs)
    plt_acc = plot_accuracy_per_epoch(epoch_accuracy, num_epochs)

    # log metrics
    run.log("accuracy", float(accuracy))
    run.log("learning_rate", learning_rate)
    run.log("num_epochs", num_epochs)
    run.log("batch_size", batch_size)
    run.log("hidden_size", hidden_size)
    run.log("total_words", total_words)
    run.log_image("Loss grapgh "+str(accuracy), plot=plt_loss)
    run.log_image("Accuracy graph "+str(accuracy), plot=plt_acc)

    # create outputs folder if not exists
    OUTPUTSFOLDER = "outputs/models"
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)

    if opts.fullmodel is True:
        # preproces data example
        y = np.empty([1, 148005])
        dummy_input = Variable(torch.FloatTensor(y))
        model_name = "net.onnx"
        pickle_name = "word2index"
        filename = os.path.join(OUTPUTSFOLDER, model_name)
        file = os.path.join(OUTPUTSFOLDER, pickle_name)
        outfile = open(file, 'wb')
        pickle.dump(word2index, outfile)
        outfile.close()

        torch.onnx.export(net, dummy_input, filename)
        # Load the ONNX model
        model = onnx.load(filename)
        model = run.register_model(model_name=model_name,
                                   model_path=filename)



