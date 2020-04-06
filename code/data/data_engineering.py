
import string
import os
import argparse
from azureml.core import Run

from packages.get_data import (load_data_from_local,
                               load_data_from_azure)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default='')
parser.add_argument("--data_local",
                    type=bool,
                    default=True)
opts = parser.parse_args()

# Get run context
run = Run.get_context()

# load data from local path
if opts.data_local is True:
    data_train, data_test = load_data_from_local(opts.dataset)
else:
    data_train, data_test = load_data_from_azure(opts.dataset, run)


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
        OUTPUTSFOLDER, opts.data_set + 'train.csv')
)

# write to csv
data_test.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDER, opts.data_set + 'test.csv')
)
