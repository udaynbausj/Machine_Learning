import numpy
import scipy
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from prep_terrain_data import makeTerrainData
from sklearn.neighbors import KNeighborsClassifier

features_train,labels_train,features_test,labels_test = makeTerrainData()

classifier = KNeighborsClassifier()
classifier.fit(features_train,labels_train)

pred = classifier.predict(features_test)

print(classifier.score(features_test,labels_test))
