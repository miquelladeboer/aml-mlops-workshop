import os
import pandas as pd


def load_data(opts):
    data_train = pd.read_csv(os.path.join(opts.data_folder_train))
    data_test = pd.read_csv(os.path.join(opts.data_folder_test))
    return(data_train, data_test)


