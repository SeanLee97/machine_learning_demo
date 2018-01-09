# !/usr/bin/env python
# -*-coding: utf-8 -*-

'''
Softmax回归模型是logistic回归模型在多分类问题上的推广
'''
import math
import pandas as pd
import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Softmax(object):
	def __init__(self):
		self.learning_rate = 0.000001 # 学习速率
		self.max_iter = 100000      # 最大迭代次数
		self.weight_lambda = 0.01   # 衰退权重
	def get_e(self, x, l):
		theta_l = self.w[l]
		product = np.dot(theta_l, x)
		return math.exp(product)
	def get_p(self, x, j):
		dividend = self.get_e(x, j)
		divisor = sum([self.get_e(x, i) for i in range(self.k)])
		return dividend/divisor
	def get_partial_derivative(self, x, y, j):
		first = int(y==j)
		second = self.get_p(x, j)
		return -x*(first-second)+self.weight_lambda*self.w[j]
	def _predict(self, x):
		result = np.dot(self.w, x)
		row, col = result.shape
		position = np.argmax(result) # 返回最大值所在的列
		m, n = divmod(position, col)
		return m
	def predict(self, features):
		labels = []
		for feature in features:
			x = list(feature)
			x.append(1)
			x = np.matrix(x)
			x = np.transpose(x)
			labels.append(self._predict(x))
		return labels

	def train(self, features, labels):
		self.k = len(set(labels))
		self.w = np.zeros((self.k, len(features[0])+1))
		times = 0
		while times<self.max_iter:
			print("loop %d" % times)
			times += 1
			index = random.randint(0, len(labels) -1)
			
			x = features[index]
			y = labels[index]
			x = list(x)
			x.append(1.0)
			x = np.array(x)
			derivatives = [self.get_partial_derivative(x, y, j) for j in range(self.k)]
			for j in range(self.k):
				self.w[j] -= self.learning_rate*derivatives[j]

def main():
	raw_data = pd.read_csv('../corpus/train.csv', header = 0)
	data = raw_data.values
	#print(raw_data)
	#print(data)
	
	imgs = data[0::, 1::]
	labels = data[::, 0]
	train_data, test_data, train_labels, test_labels = train_test_split(imgs, labels, test_size = 1/3)
	#print(test_labels)
	model = Softmax()
	print("start to train...")
	model.train(train_data, train_labels)
	
	print("start to test...")
	predict = model.predict(test_data)
	score = accuracy_score(test_labels, predict)
	print("Accuracy: " + str(score))
	
	
if __name__ == '__main__':	
	main()
