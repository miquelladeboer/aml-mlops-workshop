import pandas as pd
import os


# we only want to perform data engineering and 
# profile check on the new data that comes in
def load_data(opts):
    try:
        subsubpath = opts.data_folder_train + '/workspaceblobstore'
        dir_list_1 = os.listdir(subsubpath)
        subpath = opts.data_folder_train + '/workspaceblobstore/' + dir_list_1[0]
        df = []
        dir_list = os.listdir(
            subpath
            )
        for week in dir_list:
            print("weeks in dataset:", week)
            sub_list = os.listdir(subpath + '/' + week)
            path = os.path.join(subpath, str(week) + '/' + sub_list[0])
            dataset_train = pd.read_csv(os.path.join(path), lineterminator='\n')
            df.append(dataset_train)
        data_train = pd.concat(df, ignore_index=True)
    except FileNotFoundError:
        print("one week present")
        subsubpath = opts.data_folder_train
        print(subsubpath)
        dir_list_1 = os.listdir(subsubpath)
        print(dir_list_1)
        df = []
        for csvfile in dir_list_1:
            path = subsubpath + '/' + csvfile
            dataset_train = pd.read_csv(os.path.join(path), lineterminator='\n')
            df.append(dataset_train)
        data_train = pd.concat(df, ignore_index=True)

    try:    
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
    except FileNotFoundError:
        print("one week present")
        subsubpath = opts.data_folder_test
        dir_list_1 = os.listdir(subsubpath)
        df = []
        for csvfile in dir_list_1:
            path = subsubpath + '/' + dir_list_1[0]
            dataset_test = pd.read_csv(os.path.join(path), lineterminator='\n')
            df.append(dataset_test)
        data_test = pd.concat(df, ignore_index=True)
    return(data_train, data_test)
    