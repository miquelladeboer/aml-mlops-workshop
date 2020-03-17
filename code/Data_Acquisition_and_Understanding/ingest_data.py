# Pre-processes SKLearn sample data
# Ingest the data into an Azure ML Datastore for training
import pandas as pd
import time
import os
from sklearn.datasets import fetch_20newsgroups
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication

# Define newsgroup categories to be downloaded to generate sample dataset
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

for data_split in ['train', 'test']:
    # retrieve newsgroup data
    newsgroupdata = fetch_20newsgroups(
        subset=data_split,
        categories=categories,
        shuffle=True,
        random_state=42
    )

    # construct pandas data frame from loaded sklearn newsgroup data
    df = pd.DataFrame({
        'text': newsgroupdata.data,
        'target': newsgroupdata.target
    })

    print('data loaded')

    # pre-process:
    # remove line breaks
    # replace target index by newsgroup name
    target_names = newsgroupdata.target_names
    df.target = df.target.apply(lambda x: target_names[x])
    df.text = df.text.replace('\n', ' ', regex=True)

    print(df.head(5))

    # retrieve path of current dir
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # create temp folder if not exists
    if not os.path.exists(os.path.join(curr_dir, 'tmp', data_split)):
        os.makedirs(os.path.join(curr_dir, 'tmp', data_split))

    # write to csv
    df.to_csv(
        path_or_buf=os.path.join(
            curr_dir, 'tmp', data_split,
            '{}.csv'.format(int(time.time()))  # unique file name
        ),
        index=False,
        encoding="utf-8",
        line_terminator='\n'
    )
