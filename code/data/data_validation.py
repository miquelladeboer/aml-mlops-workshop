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
import argparse
import os
import pandas as pd
import numpy
import sys
import numpy as np
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_folder_train',
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "outputs/raw_data/raw_subset_train.csv",
    ),
    dest='data_folder_train',
    help='data folder mounting point'
)
parser.add_argument(
    '--data_folder_test',
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "outputs/raw_data/raw_subset_test.csv",
    ),
    dest='data_folder_test',
    help='data folder mounting point'
)
parser.add_argument(
    '--profile_folder',
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../..",
        "outputs/baseline_profile/baseline_profile_2020-04-21.csv",
    ),
    dest='profile_folder',
    help='profile folder mounting point'
)
opts = parser.parse_args()

# Get run context
run = Run.get_context()

nltk.download('stopwords')
stop = stopwords.words('english')

# load datasets
data_train = pd.read_csv(os.path.join(opts.data_folder_train))
data_test = pd.read_csv(os.path.join(opts.data_folder_test))

# Load baseline profile from blob
int_profile = pd.read_csv(os.path.join(opts.profile_folder))
baseline_profile = int_profile.drop('Unnamed: 0', 1)

profile_summary = []

# check data types
if data_train.text.dtype != object:
    print("Error: Training data column Text is not of type Object")
    sys.exit()

if data_test.text.dtype != object:
    print("Error: Test data column Text is not of type Object")
    sys.exit()

if data_train.target.dtype != numpy.int64:
    print("Error: Training data column Target is not of type int64")
    sys.exit()

if data_train.target.dtype != numpy.int64:
    print("Error: Training data column Test is not of type int64")
    sys.exit()

# check for na/non values
for check in data_train.isnull().all():
    if check is True:
        print("Error: Training data column" + check + "contains na/non values")
        sys.exit()
for check in data_test.isnull().all():
    if check is True:
        print("Error: Test data column" + check + "contains na/non values")
        sys.exit()

# check for right language
lang_train = np.array([])
lang_test = np.array([])
for i in range(0, len(data_train)):
    lang_train = np.append(lang_train, detect(data_train.text.values[i]))

for i in range(0, len(data_test)):
    lang_test = np.append(lang_test, detect(data_test.text.values[i]))

test2 = lang_train == 'en'
test1 = lang_test == 'en'

if test1.all() is False:
    print("Warning: There might be non english text in training data")
    warning = ["Warning: There might be non english text in training data"]
    profile_summary.append(warning)
if test2.all() is False:
    print("Warning: There might be non english text in test data")
    warning = ["Warning: There might be non english text in test data"]
    profile_summary.append(warning)

# check if classes are biased
classes = data_train.target.value_counts(normalize=True)
train_mean = classes.mean()
train_std = classes.std()
if train_std > 0.05:
    print("Warning: CLasses might be imbalanced in training data")
    warning = ["Warning: CLasses might be imbalanced in training data"]
    profile_summary.append(warning)

classes = data_test.target.value_counts(normalize=True)
test_mean = classes.mean()
test_std = classes.std()
if test_std > 0.05:
    print("Warning: CLasses might be imbalanced in test data")
    warning = ["Warning: CLasses might be imbalanced in test data"]
    profile_summary.append(warning)

# test mean and std statistics against baseline
mean = baseline_profile['mean of classes'][0]
std = baseline_profile['standard deviation of classes'][0]
if (mean - (0.05*mean)) <= train_mean <= (mean + (0.05*mean)):
    print("Warning: The classes mean might be drifting")
    warning = ["Warning: The classes mean might be drifting"]
    profile_summary.append(warning)
if (std - (0.05*std)) <= train_std <= (std + (0.05*std)):
    print("Warning: The standard deviation of classes might be drifting")
    warning = ["Warning: The standard deviation of classes might be drifting"]
    profile_summary.append(warning)


# test average word length
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))


data_avg = data_train['text'].apply(lambda x: avg_word(x)).mean()
avg = baseline_profile['average word length'][0]
if (avg - (0.05*avg)) <= data_avg <= (avg + (0.05*avg)):
    print("Warning: The average word length might be drifting")
    warning = ["Warning: The average word length might be drifting"]
    profile_summary.append(warning)

# test average number of stopwords
stopwords_avg = data_train['text'].apply(
    lambda x: len([x for x in x.split() if x in stop])
    ).mean()
stop_avg = baseline_profile['average number of stopwords'][0]
if (
 stop_avg - (0.05*stop_avg)) <= stopwords_avg <= (stop_avg + (0.05*stop_avg)):
    print("Warning: The average number of stop words might be drifting")
    warning = ["Warning: The average number of stop words might be drifting"]
    profile_summary.append(warning)

# test average sentiment
sent_avg = baseline_profile['average sentiment'][0]
sentiment = data_train['text'].apply(
    lambda x: TextBlob(x).sentiment[0]
    ).mean()
if (sent_avg - (0.05*sent_avg)) <= (sent_avg + (0.05*sent_avg)):
    print("Warning: The average sentiment might be drifting")
    warning = ["Warning: The average sentiment might be drifting"]
    profile_summary.append(warning)

run.log_list(
    "Data drift warnings",
    profile_summary,
    description='List of possible data drift detected in the new dataset compared to the baseline profile'
)
