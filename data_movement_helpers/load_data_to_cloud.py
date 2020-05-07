import os
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import uuid
from datetime import date
from sklearn.model_selection import train_test_split

weekNumber = str(date.today().isocalendar()[1]+7)
print(weekNumber)

OUTPUTSFOLDERtrain = "outputs/cloud_data_train/" + weekNumber
OUTPUTSFOLDERtest = "outputs/cloud_data_test/" + weekNumber
OUTPUTSFOLDERtrainsubset = "outputs/cloud_data_train_subset/" + weekNumber
OUTPUTSFOLDERtestsubset = "outputs/cloud_data_test_subset/" + weekNumber


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

# retrieve newsgroup data
newsgroupdata = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
)    

# construct pandas data frame from loaded sklearn newsgroup data
df = pd.DataFrame({
    'text': newsgroupdata.data,
    'target': newsgroupdata.target
})

#df1 = df.sample(n=3000)

# pre-process:
# remove line breaks
df.text = df.text.replace('\n', ' ', regex=True)

x_train, x_test = train_test_split(df, test_size=0.2)
# create random subsample to hyper tuning (50% of data)
x_train_subset = x_train.sample(frac=0.5)
x_test_subset = x_test.sample(frac=0.5)

# write to csv
x_train_subset.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDERtrainsubset,
        'raw_subset_' + "train" + '_' + str(uuid.uuid1()) +
        '.csv')
)

x_test_subset.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDERtestsubset,
        'raw_subset_' + "test" + '_' + str(uuid.uuid1()) +
        '.csv')
)

# write to csv
x_train.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDERtrain, 'raw_' + 'train' + '_' + str(uuid.uuid1()) +
        '.csv')
)

# write to csv
x_test.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDERtest, 'raw_' + 'test' + '_' + str(uuid.uuid1()) +
        '.csv')
)

# upload files
datastore.upload(
    src_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        OUTPUTSFOLDERtrain
    ),
    target_path="/" + 'raw_' + 'train' + '/' + weekNumber,
    overwrite=True,
    show_progress=True
)

# upload files
datastore.upload(
    src_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        OUTPUTSFOLDERtest
    ),
    target_path="/" + 'raw_' + 'test' + '/' + weekNumber,
    overwrite=True,
    show_progress=True
)

# upload files
datastore.upload(
    src_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        OUTPUTSFOLDERtrainsubset
    ),
    target_path="/" + 'raw_subset_' + "train" + '/' + weekNumber,
    overwrite=True,
    show_progress=True
)

# upload files
datastore.upload(
    src_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        OUTPUTSFOLDERtestsubset
    ),
    target_path="/" + 'raw_subset_' + "test" + '/' + weekNumber,
    overwrite=True,
    show_progress=True
)
