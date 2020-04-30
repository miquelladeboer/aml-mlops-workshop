import pandas as pd
import os


def load_data(opts):
    subsubpath = opts.data_folder_train + '/workspaceblobstore'
    dir_list_1 = os.listdir(subsubpath)
    subpath = opts.data_folder_train + '/workspaceblobstore/' + dir_list_1[0]
    dir_list = os.listdir(
        subpath
        )
    week = dir_list[-1]
    sub_list = os.listdir(subpath + '/' + week)
    path = os.path.join(subpath, str(week) + '/' + sub_list[0])
    data_train = pd.read_csv(os.path.join(path))

    subsubpath = opts.data_folder_test + '/workspaceblobstore'
    dir_list_1 = os.listdir(subsubpath)
    subpath = opts.data_folder_test + '/workspaceblobstore/' + dir_list_1[0]
    dir_list = os.listdir(
        subpath
        )
    week = dir_list[-1]
    sub_list = os.listdir(subpath + '/' + week)
    path = os.path.join(subpath, str(week) + '/' + sub_list[0])
    data_test = pd.read_csv(os.path.join(path))
    return(data_train, data_test)
