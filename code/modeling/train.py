import logging
import argparse
import numpy as np
import os
from azureml.core import Run
import json
import pandas as pd

from packages.get_data import (load_data)
from packages.sklearn import (Model_choice,
                              fit_sklearn,
                              pandas_to_numpy,
                              vectorizer)
from packages.plots import (plot_auc, plot_loss_per_epoch,
                            plot_accuracy_per_epoch,
                            plot_confusion_matrix,
                            plot_confusion_matrix_abs)

from sklearn.externals import joblib

# Get run context
run = Run.get_context()

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--models",
                    type=str,
                    default='sklearnmodels')
parser.add_argument("--fullmodel",
                    type=str,
                    default='no')
parser.add_argument('--data_folder_train',
                    type=str,
                    dest='data_folder_train',
                    help='data folder mounting point',
                    default=os.path.join(
                     os.path.dirname(os.path.realpath(__file__)),
                     "../..",
                     "outputs/prepared_data/subset_train.csv",
                    )
                    )
parser.add_argument('--data_folder_test',
                    type=str,
                    dest='data_folder_test',
                    help='data folder mounting point',
                    default=os.path.join(
                     os.path.dirname(os.path.realpath(__file__)),
                     "../..",
                     "outputs/prepared_data/subset_test.csv",
                    ))
parser.add_argument('--savemodel',
                    type=str)
parser.add_argument('--pipeline',
                    type=str,
                    default='no')
parser.add_argument("--output_train",
                    type=str)
parser.add_argument("--output_test",
                    type=str)

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

if not (opts.output_train is None):
    sub_train = os.listdir(opts.output_train)
    sub = ""
    for ele in sub_train:
        sub += ele
    opts.data_folder_train = opts.output_train + '/' + sub
    sub_test = os.listdir(opts.output_test)
    sub = ""
    for ele in sub_test:
        sub += ele
    opts.data_folder_test = opts.output_test + '/' + sub

data_train, data_test = load_data(opts)
print(data_train.columns.values)
print(data_train)
data_train.columns.values[-1] = 'target'
data_test.columns.values[-1] = 'target'

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
         precision, recall, f1, fpr, tpr, roc_auc,
         disp, cm, classes) = fit_sklearn(
            clf, X_train, X_test, y_train, y_test)

        # child runs
        if opts.models == 'sklearnmodels':
            # create a child run for Azure ML logging
            child_run = run.child_run(name=name)
            child_run.log("accuracy", float(accuracy))
            child_run.log("balanced accuracy", float(balanced_accuracy))
            child_run.log("F1 score", float(f1))
            child_run.log("precision", float(precision))
            child_run.log("recall", float(recall))
            plot3 = plot_auc(fpr, tpr, roc_auc)
            child_run.log_image("AUC  "+name, plot=plot3)
            child_run.log_confusion_matrix(name="confusion matrix " + name,
                                           value=disp)
            plot = plot_confusion_matrix(cm, target_names=classes)
            child_run.log_image("normalized confusion matrix " + name,
                                plot=plot)
            plot1 = plot_confusion_matrix_abs(cm, target_names=classes)
            child_run.log_image("absolute confusion matrix " + name,
                                plot=plot1)
            child_run.complete()

        # log score to AML
        run.log("accuracy", float(accuracy))
        run.log("balanced accuracy", float(balanced_accuracy))
        run.log("F1 score", float(f1))
        run.log("precision", float(precision))
        run.log("recall", float(recall))
        plot3 = plot_auc(fpr, tpr, roc_auc)
        run.log_image("AUC "+name, plot=plot3)
        run.log_confusion_matrix(name="confusion matrix " + name,
                                 value=disp)
        plot = plot_confusion_matrix(cm, target_names=classes)
        run.log_image("normalzied confusion matrix " + name, plot=plot)
        plot1 = plot_confusion_matrix_abs(cm, target_names=classes)
        run.log_image("absolute confusion matrix " + name, plot=plot1)

        # write model artifact to AML
        model_name = "model" + str(name) + ".pkl"
        filename = "outputs/models" + model_name
        joblib.dump(value=clf, filename=filename)
        run.upload_file(name=model_name, path_or_stream=filename)

else:
    import torch.nn as nn
    from torch.autograd import Variable
    import torch
    import pickle
    import onnx
    from packages.deeplearning import (index_words,
                                       get_word_2_index,
                                       get_hyperparameters,
                                       OurNet, train_model,
                                       test_model)

    try:
        with open(os.environ.get("AZUREML_DATAREFERENCE_metrics_data")) as f:
            metrics_output_result = f.read()
            deserialized_metrics_output = json.loads(metrics_output_result)
            df = pd.DataFrame(deserialized_metrics_output)
            df = df.T

            for column in df:
                df[column] = (
                    df[column]
                    .astype(str)
                    .str
                    .strip("[]")
                    .astype(float)
                )

            runID = df.accuracy.idxmax()

            parameters = df.loc[runID].iloc[1:]
            opts.learning_rate = parameters.learning_rate
            opts.num_epochs = int(parameters.num_epochs)
            opts.batch_size = int(parameters.batch_size)
            opts.hidden_size = int(parameters.hidden_size)

    except IOError:
        print("No file present")
    except TypeError:
        print("No file present")

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
    run.log_image("Loss grapgh "+str(accuracy), plot=plt_loss)
    plt_acc = plot_accuracy_per_epoch(epoch_accuracy, num_epochs)
    run.log_image("Accuracy graph "+str(accuracy), plot=plt_acc)

    # log metrics
    run.log("accuracy", float(accuracy))
    run.log("learning_rate", learning_rate)
    run.log("num_epochs", num_epochs)
    run.log("batch_size", batch_size)
    run.log("hidden_size", hidden_size)
    run.log("total_words", total_words)

    # create outputs folder if not exists
    OUTPUTSFOLDER = "outputs/models"
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)

    if opts.fullmodel == 'yes':
        # preproces data example
        y = np.empty([1, total_words])
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
     
        if not (opts.savemodel is None):
            os.makedirs(opts.savemodel, exist_ok=True)
            path = opts.savemodel + "/" + pickle_name
            path2 = opts.savemodel + "/" + model_name
            outfile = open(path, 'wb')
            pickle.dump(word2index, outfile)
            outfile.close()
            model = onnx.load(filename)
            joblib.dump(value=model, filename=path2)

