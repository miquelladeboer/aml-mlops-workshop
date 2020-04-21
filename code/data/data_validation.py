"""
Script for data validation
-----------------------------
- check input types / clumn names / string size ect
- check basic statistics
- SME knowledge data
- compare new data with old data
- check balance of classes
- check balance of classes is still valid with new data
"""
# import packages
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
opts = parser.parse_args()

# load datasets
data_train, data_test = load_data(opts)

# Load baseline profile from blob


# check for non values
