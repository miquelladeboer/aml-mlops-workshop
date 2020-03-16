import logging
import numpy as np
from optparse import OptionParser
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from azureml.core import Run

# Get run context

run = Run.get_context()

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()

argv = [] 
sys.argv[1:]
(opts, args) = op.parse_args(argv)

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


print("Loading 20 newsgroups dataset for categories:")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

# Extracting features from the training data using a sparse vectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)

# Extracting features from the test data using the same vectorizer"
X_test = vectorizer.transform(data_test.data)

# mapping from integer feature name to original token string

feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

def benchmark(clf, name=""):
    """benchmark classifier performance"""

    # train a model
    print("\nTraining run with algorithm \n{}".format(clf))
    clf.fit(X_train, y_train)

    # evaluate on test set
    pred = clf.predict(X_test)   
    score =  metrics.accuracy_score(y_test, pred)

    # log score to AML
    run.log("accuracy", float(score))

    # write model artifact to AML
    model_name = "model" + str(name) + ".pkl"
    filename = "outputs/" + model_name
    run.upload_file(name=model_name, path_or_stream=filename)

    clf_descr = str(clf).split('(')[0]
    print("Accuracy  %0.3f" % score)
    return clf_descr, score


# Run benchmark and collect results with multiple classifiers
clf = RandomForestClassifier()
name =  "Random forest"

benchmark(clf, name)

run.complete()