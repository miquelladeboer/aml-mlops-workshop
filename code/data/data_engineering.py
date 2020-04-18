
import string
import os
import argparse
from azureml.core import Run
from azureml.core import Datastore

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
                    type=str,
                    default="yes")
parser.add_argument('--local',
                    type=str,
                    default='yes')
parser.add_argument("--output_train",
                    type=str)
parser.add_argument("--output_test",
                    type=str)

opts = parser.parse_args()

# Get run context
run = Run.get_context()

if opts.local == 'yes':
    opts.data_folder_train = os.path.join(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../..",
                    "outputs/raw_data/", "raw_subset_train.csv",
                    ))
    opts.data_folder_test = os.path.join(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../..",
                    "outputs/raw_data/", "raw_subset_test.csv",
                    ))

# Load data
data_train, data_test = load_data(opts)

if opts.subset == "yes":
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

if opts.local == 'yes':
    OUTPUTSFOLDER = "C:/Users/mideboer.EUROPE/Documents/GitHub/aml-mlops-workshop/outputs/prepared_data"
# Save to local file
else:
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

if opts.local == "no":
    datastore_name = 'workspaceblobstore'

    workspace = run.experiment.workspace
    # get existing ML workspace
    # retrieve an existing datastore in the workspace by name
    datastore = Datastore.get(workspace, datastore_name)

    # upload files
    datastore.upload(
        src_dir=os.path.join(
            OUTPUTSFOLDER
        ),
        target_path=None,
        overwrite=True,
        show_progress=True
    )

    if not (opts.output_train is None):
        os.makedirs(opts.output_train, exist_ok=True)
        path = opts.output_train + "/" + dataset + 'test.csv'
        write_df = data_train.to_csv(path)

    if not (opts.output_test is None):
        os.makedirs(opts.output_test, exist_ok=True)
        path = opts.output_test + "/" + dataset + 'test.csv'
        write_df = data_test.to_csv(path)
