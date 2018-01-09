# !/usr/bin/env python
# -*- coding: utf-8 -*-

# 交叉验证来检视过拟合
from sklearn.model_selection import learning_curve # 学习曲线模块
from sklearn.datasets import load_digits
from sklearn.svm import SVC # support vector classifier
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
Y = digits.target

#print('X.shape', X.shape)

train_sizes, train_loss, test_loss = learning_curve(
	SVC(gamma=0.001), X, Y, cv=10, scoring='neg_mean_squared_error',
	train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
)

print('train_size:', train_sizes)
print('train_loss:', train_loss)
train_loss_mean = -np.mean(train_loss, axis=1) #
test_loss_mean = -np.mean(test_loss, axis=1)
print('train_loss_mean: ', train_loss_mean)

plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross-validation')
plt.xlabel('Training examples')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
