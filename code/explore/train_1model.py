import logging
from optparse import OptionParser
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Optional argument that you want to submit to the script can eb put here
op = OptionParser()

# Transform parser
argv = []
sys.argv[1:]
(opts, args) = op.parse_args(argv)

# choose categories to extract from the 20newsgroup data from sklearn
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# Load the data from sklearn 20newsgroups
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

# Extracting features from the training data using a sparse vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True,
                             max_df=0.5,
                             stop_words='english')

# Extracting features from the train and test data using the vectorizer"
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)


def benchmark(clf, name=""):
    """benchmark classifier performance"""

    # train a model
    print("\nTraining run with algorithm \n{}".format(clf))
    clf.fit(X_train, y_train)

    # evaluate on test set
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)

    # Print model accuracy
    clf_descr = str(clf).split('(')[0]
    print("Accuracy  %0.3f" % score)

    return clf_descr, score


# Select model to benchmark (In the lab we choose for Random Forest,
# but any classification model from sklearn can be chosen)
clf = RandomForestClassifier()
name = "Random forest"

# Run benchmark and collect results from the selected mdodel
benchmark(clf, name)
