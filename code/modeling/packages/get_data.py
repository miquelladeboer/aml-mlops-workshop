import os
import pandas as pd


def load_data(opts):
    dataset_train = pd.read_csv(os.path.join(opts.data_folder_train),
                                lineterminator='\n')
    dataset_test = pd.read_csv(os.path.join(opts.data_folder_test),
                               lineterminator='\n')

    return(dataset_train, dataset_test)


