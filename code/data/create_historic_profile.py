"""
Script the creation of the historic  profile
We use the historic profiles to see and monitor 
how our data is changing over time 
In this senario we assume that data is loaded every day
This is the profile we automatically train (every day) from
a trigger from cloud with cloud data (pipeline-step)
-----------------------------
- report basic statistics
- SME knowledge data
- use report for over-time changes as well
"""
# load packeges.
import pandas as pd
import string
import nltk
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from textblob import TextBlob
from datetime import date

nltk.download('stopwords')
stop = stopwords.words('english')

today = date.today()

# import data from local to create baseline profile
data = pd.read_csv("C:/Users/mideboer.EUROPE/Documents/GitHub/aml-mlops-workshop/outputs/raw_data/raw_subset_train.csv")

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
                                      'average sentiment'], index=today)

# get words first
data_clean = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df2 = pd.DataFrame()
df1 = pd.DataFrame()

# create dataframe


for classes in range(0, 4):
    vocab = Counter()
    df_train = data_clean[data.target == classes]
    for text in df_train:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    word1 = 'word' + str(classes)
    count = 'count' + str(classes)
    idf = 'idf' + str(classes)
    tfidf = 'tfidf' + str(classes)

    df = pd.DataFrame(list(vocab.items()), columns=[word1, count])

    for i, word in enumerate(df[word1]):
        df.loc[i, idf] = np.log(df_train.shape[0] /
                                (len(df_train[df_train.str.contains(word)])))

    array = df[count] * df[idf]
    df[tfidf] = array

    df1 = df.sort_values(tfidf, ascending=False, ignore_index=True)
    df2[word1] = df1[word1]
    df2[tfidf] = df1[tfidf]

most_importantwords = df2.iloc[0:20]
most_importantwords.columns = ['alt.atheism', 'tfidf1',
                               'talk.religion.misc', 'tfidf2',
                               'comp.graphics', 'tfidf3',
                               'sci.space', 'tfidf4']
print(profile)