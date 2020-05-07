"""
Scripts for downloading sample data from SKLearn to you local machine
-----------------------------
- download 4 categories from sklearn data to your local folder.
"""
import pandas as pd
import os
from sklearn.datasets import fetch_20newsgroups

# Set outputrs folder to outputs/raw_data
OUTPUTSFOLDER = "outputs/raw_data"

# create outputs folder if not exists, create the folder
if not os.path.exists(OUTPUTSFOLDER):
    os.makedirs(OUTPUTSFOLDER)

# Define newsgroup categories to be downloaded to generate sample dataset
# In this example, we choose 4, categories, for all categories check the page
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
# The data contains text messages that are classified by their category
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# We split the data into train and test data
for data_split in ['train', 'test']:
    # retrieve newsgroup data
    newsgroupdata = fetch_20newsgroups(
        subset=data_split,
        categories=categories,
        shuffle=True,
        random_state=42
    )

    # construct pandas data frame from loaded sklearn newsgroup data
    # text column will contain the body of text message, target column
    # contains the categories.
    df = pd.DataFrame({
        'text': newsgroupdata.data,
        'target': newsgroupdata.target
    })

    # pre-process:
    # remove line breaks
    df.text = df.text.replace('\n', ' ', regex=True)

    # create random subsample to hyper tuning (50% of data)
    df_subset = df.sample(frac=0.5)

    # write to csv to outputs folder
    df_subset.to_csv(
        path_or_buf=os.path.join(
            OUTPUTSFOLDER, 'raw_subset_' + data_split +
            '.csv')
    )