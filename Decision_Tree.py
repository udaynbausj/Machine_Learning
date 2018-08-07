from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

iris = load_iris()

clf = tree.DecisionTreeClassifier()

nb_clf = GaussianNB()

#loading and training the dataset

clf.fit(iris.data,iris.target)

nb_clf.fit(iris.data,iris.target)

#lets test now

pred = clf.predict(iris.data)
print(clf.score(iris.data,iris.target))
