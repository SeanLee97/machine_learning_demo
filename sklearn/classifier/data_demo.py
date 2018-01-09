# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = datasets.load_boston() # 加载波士顿房价
data_x = data.data
data_y = data.target

'''
print(data_x[:2, 1:])
print(data_y[:2])
'''

model = LinearRegression()
model.fit(data_x, data_y)

pred = model.predict(data_x)
print(model.coef_)
plt.scatter(pred, data_y)
plt.show()


'''
x, y = datasets.make_regression(n_samples=100, n_features = 1, n_targets=1, noise=10)
plt.scatter(x, y)
plt.show()
'''
