import logging
import argparse
import numpy as np
import os
from azureml.core import Run
import json
import pandas as pd
import pandas

from packages.sklearn import (Model_choice,
                              fit_sklearn,
                              pandas_to_numpy,
                              vectorizer)
from packages.plots import (plot_auc, plot_loss_per_epoch,
                            plot_accuracy_per_epoch,
                            plot_confusion_matrix,
                            plot_confusion_matrix_abs)
from packages.get_data import load_data
from sklearn.externals import joblib

# Get run context
run = Run.get_context()

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

parser = argparse.ArgumentParser()
# allowed arguments are: randomforest, sklearn, deeplearning
# randomforest will perform 1 run of randomforest fit
# sklearnmodels will fit 15 models from sklearn
# deeplearning will fit a neural network with pytorch
parser.add_argument(
    "--models",
    type=str,
    default='randomforest'
)
parser.add_argument(
    '--local',
    type=str,
    default='yes'
)
parser.add_argument(
    "--fullmodel",
    type=str,
    default='no'
)
parser.add_argument(
    '--data_folder_train',
    type=str,
    dest='data_folder_train',
    help='data folder mounting point',
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "outputs/prepared_data/subset_train.csv",
    )
)
parser.add_argument(
    '--data_folder_test',
    type=str,
    dest='data_folder_test',
    help='data folder mounting point',
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "outputs/prepared_data/subset_test.csv",
    )
)
parser.add_argument(
    '--savemodel',
    type=str
)
parser.add_argument(
    "--output_train",
    type=str
)
parser.add_argument(
    "--output_test",
    type=str
)
parser.add_argument(
    "--input_train",
    type=str
)
parser.add_argument(
    "--input_test",
    type=str
)
parser.add_argument(
    "--sklearnmodel",
    type=str
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.1
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=20
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=100
)
opts = parser.parse_args()

# load in data from two different sources
# if data is pipeline data
if not (opts.input_train is None):
    data_train = pd.read_csv(
            os.path.join(
                opts.input_train + '_prepared.csv'
            ),
            lineterminator='\n'
        )
    data_test = pd.read_csv(
            os.path.join(
                opts.input_test + '_prepared.csv'
            ),
            lineterminator='\n'
        )
else:
    # load data from local
    try:
        data_train = pd.read_csv(
                os.path.join(
                    opts.data_folder_train
                ),
                lineterminator='\n'
            )
        data_test = pd.read_csv(
                os.path.join(
                    opts.data_folder_test
                ),
                lineterminator='\n'
            )
    except pandas.errors.ParserError:
        data_train, data_test = load_data(opts)

# set right name for target variable
data_train.columns.values[-1] = 'target'
data_test.columns.values[-1] = 'target'

# if we want to fit either sklearnmodels or randomforest
if opts.models != 'deeplearning':
    # transform data from pandas to numpy
    X_train, X_test, y_train, y_test = pandas_to_numpy(data_train, data_test)
    # vectorize data
    X_train, X_test = vectorizer(X_train, X_test)
    # Get models from sklearn
    # can be 15 models or 1 randomforest
    models = Model_choice(opts.models)

    # Run benchmark and collect results with multiple classifiers
    # for evert model in Model_choice
    for i in range(models.number_of_models()):
        # fit model
        clf, name = models.select_model(i)

        # get importan metrics
        (clf_descr, accuracy, balanced_accuracy,
         precision, recall, f1, fpr, tpr, roc_auc,
         disp, cm, classes) = fit_sklearn(
            clf, X_train,
            X_test,
            y_train,
            y_test
        )

        # create child runs if we have many models
        if opts.models == 'sklearnmodels':
            # create a child run for Azure ML logging
            child_run = run.child_run(name=name, outputs="outputs/models")
            # start logging all metrics to aml
            child_run.log(
                "accuracy",
                float(accuracy)
            )
            child_run.log(
                "balanced accuracy",
                float(balanced_accuracy)
            )
            child_run.log(
                "F1 score",
                float(f1)
            )
            child_run.log(
                "precision",
                float(precision)
            )
            child_run.log(
                "recall",
                float(recall)
            )
            plot3 = plot_auc(
                fpr,
                tpr,
                roc_auc
            )
            child_run.log_image(
                "AUC  "+name,
                plot=plot3
            )
            child_run.log_confusion_matrix(
                name="confusion matrix " + name,
                value=disp
            )
            plot = plot_confusion_matrix(
                cm,
                target_names=classes
            )
            child_run.log_image(
                "normalized confusion matrix " + name,
                plot=plot
            )
            plot1 = plot_confusion_matrix_abs(
                cm,
                target_names=classes
            )
            child_run.log_image(
                "absolute confusion matrix " + name,
                plot=plot1
            )

            model_name = "model" + str(name) + ".pkl"
            filename = "outputs/models" + model_name
            joblib.dump(
                value=clf,
                filename=filename
            )
            child_run.upload_file(
                name=model_name,
                path_or_stream=filename
            )
            child_run.complete()

        # log score to AML
        run.log(
            "accuracy",
            float(accuracy)
        )
        run.log(
            "balanced accuracy",
            float(balanced_accuracy)
        )
        run.log(
            "F1 score",
            float(f1)
        )
        run.log(
            "precision",
            float(precision)
        )
        run.log(
            "recall",
            float(recall)
        )
        plot3 = plot_auc(
            fpr,
            tpr,
            roc_auc
        )
        run.log_image(
            "AUC "+name,
            plot=plot3
        )
        run.log_confusion_matrix(
            name="confusion matrix " + name,
            value=disp
        )
        plot = plot_confusion_matrix(
            cm,
            target_names=classes
        )
        run.log_image(
            "normalzied confusion matrix " + name,
            plot=plot
        )
        plot1 = plot_confusion_matrix_abs(
            cm,
            target_names=classes
        )
        run.log_image(
            "absolute confusion matrix " + name,
            plot=plot1
        )

        # write model artifact to AML
        model_name = "model" + str(name) + ".pkl"
        filename = "outputs/models" + model_name
        joblib.dump(
            value=clf,
            filename=filename
        )
        run.upload_file(
            name=model_name,
            path_or_stream=filename
        )
    # if we are not running local, we can retrieve best run
    if opts.local == "no":
        if opts.models == 'sklearnmodels':
            max_accuracy_runid = None
            max_accuracy = None
            modelfile = None
            best_run = None
            for childrun in run.get_children():
                run_metrics = childrun.get_metrics()
                run_details = childrun.get_details()
                run_files = child_run.get_file_names
                run_accuracy = run_metrics["accuracy"]
                run_id = run_details["runId"]

                if max_accuracy is None:
                    max_accuracy = run_accuracy
                    max_accuracy_runid = run_id
                    best_run = childrun
                else:
                    if run_accuracy > max_accuracy:
                        max_accuracy = run_accuracy
                        max_accuracy_runid = run_id
                        best_run = child_run

            print("Best run_id: " + max_accuracy_runid)
            print("Best run_id accuracy: " + str(max_accuracy))
            # all_files = best_run.get_file_names()
            # sub = '.pkl'
            # bestfilename = [i for i in all_files if sub in i]
            # files = bestfilename[0]
            # print(files)
            # if not (opts.sklearnmodel is None):
            #     best_run.download_file(
            #         name=files,
            #         output_file_path=opts.sklearnmodel
            #    )

else:
    # fit deeplearning model
    import torch.nn as nn
    from torch.autograd import Variable
    import torch
    import pickle
    import onnx
    from packages.deeplearning import (
        index_words,
        get_word_2_index,
        get_hyperparameters,
        OurNet, train_model,
        test_model
        )
    # if we are runnig pipeline data, we can use the hyper parameters from the
    # hyperdrive to train the model with these parameters. 
    try:
        with open(os.environ.get("AZUREML_DATAREFERENCE_metrics_data")) as f:
            # read json
            metrics_output_result = f.read()
            deserialized_metrics_output = json.loads(metrics_output_result)
            df = pd.DataFrame(deserialized_metrics_output)
            df = df.T

            # make pandas dataframe in rtight format
            for column in df:
                df[column] = (
                    df[column]
                    .astype(str)
                    .str
                    .strip("[]")
                    .astype(float)
                )

            # get run ID of best run
            runID = df.accuracy.idxmax()

            # select parameters of best run
            parameters = df.loc[runID].iloc[1:]
            print(parameters)
            # set parameters for model to the best run parameters
            opts.learning_rate = parameters.learning_rate
            opts.num_epochs = int(parameters.num_epochs)
            opts.batch_size = int(parameters.batch_size)
            opts.hidden_size = int(parameters.hidden_size)

    except IOError:
        print("No file present")
    except TypeError:
        print("No file present")

    # calculate the indexes of the words present in the dataset
    vocab, total_words = index_words(
        data_train,
        data_test
    )
    word2index = get_word_2_index(vocab)

    # set hyperparameters
    (learning_rate, num_epochs,
        batch_size, hidden_size) = get_hyperparameters(opts)
    input_size = total_words  # Words in vocab
    num_classes = len(np.unique(data_train.target)) # 4 categories in the example

    # output [max index for each item in batch, ... ,batch_size-1]
    loss = nn.CrossEntropyLoss()
    input = Variable(torch.randn(2, 5), requires_grad=True)

    target = Variable(torch.LongTensor(2).random_(5))

    output = loss(input, target)
    output.backward()

    # set structure of neural network
    net = OurNet(
        input_size,
        hidden_size,
        num_classes
    )

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate
    )

    # train neural network
    epoch_losses, epoch_accuracy, net = train_model(
        num_epochs,
        data_train,
        data_test,
        batch_size,
        criterion,
        optimizer,
        net,
        total_words,
        word2index
    )

    # get accuracy of network woth out of sample data
    accuracy = test_model(
        data_test,
        net,
        total_words,
        word2index
    )
    print("model accuracu out of sample:", accuracy)

    # plot loss per epoch
    plt_loss = plot_loss_per_epoch(
        epoch_losses,
        num_epochs
    )
    run.log_image(
        "Loss grapgh "+str(accuracy),
        plot=plt_loss
    )

    # plot accuracy per epoch
    plt_acc = plot_accuracy_per_epoch(
        epoch_accuracy,
        num_epochs
    )
    run.log_image(
        "Accuracy graph "+str(accuracy),
        plot=plt_acc
    )

    # log metrics
    run.log(
        "accuracy",
        float(accuracy)
    )
    run.log(
        "learning_rate",
        learning_rate
    )
    run.log(
        "num_epochs",
        num_epochs
    )
    run.log(
        "batch_size",
        batch_size
    )
    run.log(
        "hidden_size",
        hidden_size
    )
    run.log(
        "total_words",
        total_words
    )

    # create outputs folder if not exists
    OUTPUTSFOLDER = "outputs/models"
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)

    if opts.fullmodel == 'yes':
        # generate some small sample data
        y = np.empty([1, total_words])
        dummy_input = Variable(torch.FloatTensor(y))

        # save model to onnx with sample data and word indexes
        model_name = "net.onnx"
        pickle_name = "word2index"
        filename = os.path.join(
            OUTPUTSFOLDER,
            model_name
        )
        file = os.path.join(
            OUTPUTSFOLDER,
            pickle_name
        )
        outfile = open(file, 'wb')
        pickle.dump(
            word2index,
            outfile
        )
        outfile.close()

        torch.onnx.export(
            net,
            dummy_input,
            filename
        )
        # Load the ONNX model
        model = onnx.load(filename)
        model = run.register_model(
            model_name=model_name,
            model_path=filename
        )

        # if pipeline, generate model path as ouput pipeline data
        if not (opts.savemodel is None):
            os.makedirs(opts.savemodel, exist_ok=True)
            path = opts.savemodel + "/" + pickle_name
            path2 = opts.savemodel + "/" + model_name
            outfile = open(path, 'wb')
            pickle.dump(
                word2index,
                outfile
            )
            outfile.close()
            model = onnx.load(filename)
            joblib.dump(
                value=model,
                filename=path2
            )
