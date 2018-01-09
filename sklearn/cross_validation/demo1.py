# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score # K折交叉验证模块
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# load dataset
iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# create model
model = KNeighborsClassifier()

# fit a mdoel
model.fit(X_train, Y_train)

# evaluate
print(model.score(X_test, Y_test))

scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')  # 计算准确率
print(scores) # 将5次的预测准确率打印

print('平均准确率', scores.mean()) # 输出平均准确率

k_range = range(1, 31)
k_scores = [] # to save the scores

for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X, Y ,cv=10, scoring='accuracy')
	k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('value of k for knn')
plt.ylabel('cross-validation accuracy')
plt.show()

# scoring='mean_squared_error' 一般用于回归模型的好坏
k_range = range(1, 31)
k_scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	loss = -cross_val_score(knn, X, Y, cv=10, scoring='mean_squared_error')
	k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('values of k for knn')
plt.ylabel('cross-validation MSE')
plt.show()

# https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-3-cross-validation2/
