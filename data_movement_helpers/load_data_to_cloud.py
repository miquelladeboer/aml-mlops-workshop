import os
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import uuid 

OUTPUTSFOLDERtrain = "outputs/cloud_data_train"
OUTPUTSFOLDERtest = "outputs/cloud_data_test"
OUTPUTSFOLDERtrainsubset = "outputs/cloud_data_train_subset"
OUTPUTSFOLDERtestsubset = "outputs/cloud_data_test_subset"

datastore_name = 'workspaceblobstore'

# get existing ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)

# create outputs folder if not exists
if not os.path.exists(OUTPUTSFOLDERtrain):
        os.makedirs(OUTPUTSFOLDERtrain)

if not os.path.exists(OUTPUTSFOLDERtest):
        os.makedirs(OUTPUTSFOLDERtest)

if not os.path.exists(OUTPUTSFOLDERtrainsubset):
        os.makedirs(OUTPUTSFOLDERtrainsubset)

if not os.path.exists(OUTPUTSFOLDERtestsubset):
        os.makedirs(OUTPUTSFOLDERtestsubset)


# Define newsgroup categories to be downloaded to generate sample dataset
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

for data_split in ['train', 'test']:
    if data_split ==  'train':
        OUTPUTSFOLDER =  OUTPUTSFOLDERtrain
        OUTPUTSFOLDERSUBSET = OUTPUTSFOLDERtrainsubset
    else:
        OUTPUTSFOLDER = OUTPUTSFOLDERtest
        OUTPUTSFOLDERSUBSET = OUTPUTSFOLDERtestsubset

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
    df_subset.to_csv(
        path_or_buf=os.path.join(
            OUTPUTSFOLDERSUBSET, 'raw_subset_' + data_split + '_' + str(uuid.uuid1()) +
            '.csv')
    )

    # write to csv
    df.to_csv(
        path_or_buf=os.path.join(
            OUTPUTSFOLDER, 'raw_' + data_split + '_' + str(uuid.uuid1()) +
            '.csv')
    )


    # upload files
    datastore.upload(
        src_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../',
            OUTPUTSFOLDER
        ),
        target_path="/" + 'raw_' + data_split,
        overwrite=True,
        show_progress=True
    )

     # upload files
    datastore.upload(
        src_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../',
            OUTPUTSFOLDERSUBSET
        ),
        target_path="/" + 'raw_subset_' + data_split,
        overwrite=True,
        show_progress=True
    )
