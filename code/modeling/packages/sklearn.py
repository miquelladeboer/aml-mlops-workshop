from sklearn.metrics import (roc_curve, accuracy_score, auc,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

import numpy as np


def pandas_to_numpy(data_train, data_test):
    X_train = data_train.text
    X_test = data_test.text
    y_train = data_train.target
    y_test = data_test.target
    return(X_train, X_test, y_train, y_test)


def vectorizer(X_train, X_test):
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 max_df=0.5,
                                 stop_words='english')

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test


def fit_sklearn(clf, X_train, X_test, y_train, y_test):
    """benchmark classifier performance"""

    # train a model
    print("\nTraining run with algorithm \n{}".format(clf))
    clf.fit(X_train, y_train)

    # evaluate on test set
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    balanced_accuracy = balanced_accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')

    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, pred == i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    clf_descr = str(clf).split('(')[0]
    print("Accuracy  %0.3f" % accuracy)
    return (clf_descr, accuracy, balanced_accuracy,
            precision, recall, f1, fpr, tpr, roc_auc)


class Model_choice:
    def __init__(self, object):
        if object == 'randomforest':
            self.models = {
                (RandomForestClassifier(),
                 "Random forest")
                }
        if object == 'sklearnmodels':
            self.models = {
                (RidgeClassifier(tol=1e-2,
                                 solver="sag"),
                 "Ridge Classifier"),
                (Perceptron(max_iter=50),
                 "Perceptron"),
                (PassiveAggressiveClassifier(max_iter=50),
                 "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10),
                 "kNN"),
                (RandomForestClassifier(),
                 "Random forest"),
                (LinearSVC(penalty="l2",
                           dual=False,
                           tol=1e-3),
                 "L2 Linear SVC"),
                (LinearSVC(penalty="l1",
                           dual=False,
                           tol=1e-3),
                 "L1 Linear SVC"),
                (SGDClassifier(alpha=.0001,
                               max_iter=50,
                               penalty="l2"),
                 "L2 SGDClassifier"),
                (SGDClassifier(alpha=.0001,
                               max_iter=50,
                               penalty="l1"),
                 "L1 SGDClassifier"),
                (SGDClassifier(alpha=.0001,
                               max_iter=50,
                               penalty="elasticnet"),
                 "Elastic-Net penalty SGDClassifier"),
                (NearestCentroid(),
                 "NearestCentroid (aka Rocchio classifier)"),
                (MultinomialNB(alpha=.01),
                 "Naive Bayes MultinomialNB"),
                (BernoulliNB(alpha=.01),
                 "Naive Bayes BernoulliNB"),
                (ComplementNB(alpha=.1),
                 "Naive Bayes ComplementB"),
                (Pipeline([
                 ('feature_selection', SelectFromModel(
                        LinearSVC(penalty="l1",
                                  dual=False,
                                  tol=1e-3)
                                  )
                  ),
                    ('classification',
                     LinearSVC(penalty="l2"))]),
                 "LinearSVC with L1-based feature selection")
            }

    def number_of_models(self):
        return len(self.models)

    def select_model(self, i):
        model, name = list(self.models)[i]
        return model, name
