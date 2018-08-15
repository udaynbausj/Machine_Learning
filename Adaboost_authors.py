from sklearn.ensemble import AdaBoostClassifier

# !/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys

from sklearn.naive_bayes import GaussianNB
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

classifier = AdaBoostClassifier()

t0 = time()
classifier.fit(features_train, labels_train)

print("Training time : ", round(time() - t0, 3))

t0 = time()

pred = list(classifier.predict(features_test))

print("Predicting time : ", round(time() - t0, 3))

print(classifier.score(features_test, labels_test))
