# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# 1. processe data

# 1.1 get data
iris = load_iris()
X = iris.data
y = iris.target
# 1.2 data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. pipeline
pipe = Pipeline(steps=[
	('scaler', StandardScaler()),
	('pca', PCA(n_components=2)),
	('clf', SVC(kernel='linear'))
])

# 3. GridSearch
estimator = GridSearchCV(pipe, param_grid={'clf__C': [1., 1.3, 1.5, 2., 2.3, 2.5, 3., 3.3, 3.5]})

estimator.fit(X_train, y_train)

score = estimator.score(X_test, y_test)
print('accuracy', score)
print('best estimator', estimator.best_estimator_)
