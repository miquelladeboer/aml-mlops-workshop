import os
import pandas as pd


def load_data(opts):
    subsubpath = opts.data_folder_train + '/workspaceblobstore'
    dir_list_1 = os.listdir(subsubpath)
    subpath = opts.data_folder_train + '/workspaceblobstore/' + dir_list_1[0]
    df = []
    dir_list = os.listdir(
        subpath
        )
    for week in dir_list:
        print(week)
        sub_list = os.listdir(subpath + '/' + week)
        path = os.path.join(subpath, str(week) + '/' + sub_list[0])
        dataset_train = pd.read_csv(os.path.join(path), lineterminator='\n')
        df.append(dataset_train)
    data_train = pd.concat(df, ignore_index=True)

    subsubpath = opts.data_folder_test + '/workspaceblobstore'
    dir_list_1 = os.listdir(subsubpath)
    subpath = opts.data_folder_test + '/workspaceblobstore/' + dir_list_1[0]
    df = []
    dir_list = os.listdir(
        subpath
        )
    for week in dir_list:
        sub_list = os.listdir(subpath + '/' + week)
        path = os.path.join(subpath, str(week) + '/' + sub_list[0])
        dataset_test = pd.read_csv(os.path.join(path), lineterminator='\n')
        df.append(dataset_test)
    data_test = pd.concat(df, ignore_index=True)

    return(data_train, data_test)

