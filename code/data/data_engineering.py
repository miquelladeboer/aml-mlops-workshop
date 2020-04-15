
import string
import os
import argparse
from azureml.core import Run
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication

from packages.get_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_train',
                    type=str,
                    dest='data_folder_train',
                    help='data folder mounting point')
parser.add_argument('--data_folder_test',
                    type=str,
                    dest='data_folder_test',
                    help='data folder mounting point')
parser.add_argument('--subset',
                    default=False)
parser.add_argument('--local',
                    default=True)                    
opts = parser.parse_args()

# Get run context
run = Run.get_context()

# Load data
data_train, data_test = load_data(opts)

if opts.subset is True:
    dataset = "subset_"
else:
    dataset = ""

# make every thing lower case
data_train.text = data_train.text.apply(lambda x: x.lower())
data_test.text = data_test.text.apply(lambda x: x.lower())

# remove punctuation
translator = str.maketrans('', '', string.punctuation)
data_train.text = data_train.text.apply(
    lambda x: x.translate(translator))
data_test.text = data_test.text.apply(
    lambda x: x.translate(translator))

# remoce digits
data_train.text = data_train.text.apply(
    lambda x: x.translate(string.digits))
data_test.text = data_test.text.apply(
    lambda x: x.translate(string.digits))

# Save to local file
OUTPUTSFOLDER = "outputs/prepared_data"

# create outputs folder if not exists
if not os.path.exists(OUTPUTSFOLDER):
    os.makedirs(OUTPUTSFOLDER)

# write to csv
data_train.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDER, dataset + 'train.csv')
)

# write to csv
data_test.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDER, dataset + 'test.csv')
)

if opts.local is False:
    datastore_name = 'workspaceblobstore'

    # get existing ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())

    # retrieve an existing datastore in the workspace by name
    datastore = Datastore.get(workspace, datastore_name)

    # upload files
    datastore.upload(
        src_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../',
            OUTPUTSFOLDER
        ),
        target_path=None,
        overwrite=True,
        show_progress=True
    )
        
