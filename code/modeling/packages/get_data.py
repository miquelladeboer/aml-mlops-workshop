import os
import pandas as pd


def load_data_from_local(dataset):
    # datafolder
    DATAFOLDER = 'C:/Users/mideboer.EUROPE/Documents/GitHub/aml-mlops-workshop/outputs/prepared_data/'
    # load data from local path
    path_train = os.path.join(
                    DATAFOLDER, dataset + 'train.csv'
                    )
    path_test = os.path.join(
                    DATAFOLDER, dataset + 'test.csv'
                    )
    data_train = pd.read_csv(path_train)
    data_test = pd.read_csv(path_test)
    return(data_train, data_test)


def load_data_from_azure(dataset, run):
    dataset_train = run.input_datasets[dataset + 'train'].to_pandas_dataframe()
    dataset_test = run.input_datasets[dataset + 'test'].to_pandas_dataframe()

    # # Pre-process df for sklearn
    class data_train(object):
        def __init__(self, data, target):
            self.text = []
            self.target = []

    class data_test(object):
        def __init__(self, data, target):
            self.text = []
            self.target = []

    # convert to numpy df
    data_train.text = dataset_train.text.values
    data_test.text = dataset_test.text.values

    # convert label to int
    data_train.target = [int(value or 0)
                         for value in dataset_train.target.values]
    data_test.target = [int(value or 0)
                        for value in dataset_test.target.values]
    return(data_train, data_test)
