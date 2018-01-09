# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Classifier type.')
parser.add_argument('-t', '--type', dest='type', help='please input classifier type', default='knn')
args = parser.parse_args()

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)

if 'nbayes' == args.type.lower():
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB(0.2)
else:
	from sklearn.neighbors import KNeighborsClassifier
	clf = KNeighborsClassifier()

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

total_num = len(y_test)
correct_num = total_num - np.count_nonzero(y_test-pred)
print(correct_num/total_num)
