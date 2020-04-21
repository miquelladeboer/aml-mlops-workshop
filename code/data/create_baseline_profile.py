"""
Script the creation of the baseline profile
-----------------------------
- report basic statistics
- SME knowledge data
- use report for over-time changes as well
"""
# load packeges.
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication
import os
from datetime import date
nltk.download('stopwords')
stop = stopwords.words('english')

today = date.today()

datastore_name = 'workspaceblobstore'

# get existing ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)

# create outputs folder
OUTPUTSFOLDER = 'outputs/baseline_profile'
# create outputs folder if not exists

if not os.path.exists(OUTPUTSFOLDER):
    os.makedirs(OUTPUTSFOLDER)

# import data from local to create baseline profile
data = pd.read_csv(os.path.join(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../..",
                    "outputs/raw_data/raw_subset_train.csv",
                    )))

# clean data
data.text = data.text.apply(lambda x: x.lower())
translator = str.maketrans('', '', string.punctuation)
data.text = data.text.apply(
    lambda x: x.translate(translator))
data.text = data.text.apply(
    lambda x: x.translate(string.digits))

# Classes balance + std
classes = data.target.value_counts(normalize=True)
data_mean = classes.mean()
data_std = classes.std()


# avg word length
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))


data['avg_word'] = data['text'].apply(lambda x: avg_word(x))
mean_avg_word = data['avg_word'].mean()

# avg number of stop words
data['stopwords'] = data['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
mean_stopwords = data['stopwords'].mean()

# setiment analyse
data['sentiment'] = data['text'].apply(lambda x: TextBlob(x).sentiment[0])
mean_sentiment = data['sentiment'].mean()

# creat df for profiling
data = {'mean of classes':  [data_mean],
        'standard deviation of classes': [data_std],
        'average word length': [mean_avg_word],
        'average number of stopwords': [mean_stopwords],
        'average sentiment': [mean_stopwords]
        }

profile = pd.DataFrame(data, columns=['mean of classes',
                                      'standard deviation of classes',
                                      'average word length',
                                      'average number of stopwords',
                                      'average sentiment'])


# save profile to local file
profile.to_csv(
    path_or_buf=os.path.join(
        OUTPUTSFOLDER,
        'baseline_profile_' + str(today) + '.csv')
    )

# upload profile to blob
datastore.upload(
        src_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../',
            OUTPUTSFOLDER
        ),
        target_path="baseline_profile",
        overwrite=False,
        show_progress=True
    )
