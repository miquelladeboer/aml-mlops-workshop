# Pre-processes SKLearn sample data
# Ingest the data into an Azure ML Datastore for training
import pandas as pd
import os
from sklearn.datasets import fetch_20newsgroups

OUTPUTSFOLDER = "outputs/raw_data"

# create outputs folder if not exists
if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)


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

    # pre-process:
    # remove line breaks
    df.text = df.text.replace('\n', ' ', regex=True)

    # create random subsample to hyper tuning (50% of data)
    df_subset = df.sample(frac=0.5)

    # write to csv
    df.to_csv(
        path_or_buf=os.path.join(
            OUTPUTSFOLDER, 'raw_' + data_split +
            '.csv')
    )

    # write to csv
    df_subset.to_csv(
        path_or_buf=os.path.join(
            OUTPUTSFOLDER, 'raw_subset_' + data_split +
            '.csv')
    )