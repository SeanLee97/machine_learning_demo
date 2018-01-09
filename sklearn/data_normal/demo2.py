# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# 生成适合做Classification的数据
x, y = make_classification(
	n_samples = 300,
	n_features = 2,   # 特征数 = n_informative + n_redundant + n_repeated
	n_redundant = 0,  # 冗余数
	n_informative = 2, # 多信息特征的个数
	#n_repeated = 0,   # 重复信息(随机抽取n_informative和n_redundant的特征)
	random_state = 22,# 
	n_clusters_per_class = 1, # 某个类别有多少个cluster构成
	scale = 100 # 规模大小
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = SVC()
clf.fit(x_train, y_train)
print("数据未标准化的分类准确率:", clf.score(x_test, y_test))

print('x shape: ', x.shape) # (300, 2)
print('y shape: ', y.shape) # (300, )
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# 标准化数据
x = preprocessing.scale(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = SVC()
clf.fit(x_train, y_train)
print('标准化数据后分类准确率:', clf.score(x_test, y_test))

