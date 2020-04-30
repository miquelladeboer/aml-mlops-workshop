# Data Movement helpers
This folder contains some scripts for the initial set-up of the enviroment.
* load_data_from_web.py 

    Helper file to load raw data from sklearn   20newsgroups to your local outputsfolder: "outputs/raw_data". (Use this folder if you want to have your data locally for exploration or debugging using the raw data.)

* load_data_to_cloud.py 

    Helper file to load raw data from sklearn 20newsgroups to the default blobstorage of your workspace.

* define_dataset_raw.py

    Helper file to define the dataset that refers to the raw data in your blobstorage.

* define_dataset_prepared.py

    Helper filde to define the dataset thet refers to the prepared data in your blobstorage (note: you can only set up this after you have prepared data in your blobstorage. To do that, run first the data_engineering_submit.py with parameters, data_local = 'no' and subset = 'no' and for  data_local = 'no' and subset = 'yes')



