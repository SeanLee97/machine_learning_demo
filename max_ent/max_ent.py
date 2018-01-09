# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math, random, time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
最大熵模型
'''
class MaxEnt(object):
	def init_params(self, X, Y):
		self._X = X
		self._Y = Y
		self.get_Pxy_Px(X, Y)
		self.N = len(X)	         # 训练集大小
		self.n = len(self.Pxy)   # 对数
		self.M = 10000.0
		
		self.build_dict()
		self.get_EPxy()
	def build_dict(self):
		self.id2xy = {}
		self.xy2id = {}
		
		for i, (x,y) in enumerate(self.Pxy):
			self.id2xy[i] = (x, y)
			self.xy2id[(x, y)] = i
	def get_Pxy_Px(self, X,Y):
		self.Pxy = defaultdict(int)
		self.Px = defaultdict(int)

		for i in range(len(X)):
			_x, y = X[i], Y[i]
			self._Y.add(y)
			for x in _x:
				self.Pxy[(x, y)] += 1
				self.Px[x] += 1
	def get_EPxy(self):
		self.EPxy = defaultdict(float)
		for id in range(self.n):
			(x, y) = self.id2xy[id]
			self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

	def get_pxy(self, X, y):
		result = 0.0
		for x in X:
			if self.fxy(x, y):
				id = self.xy2id[(x, y)]
				result += self.w[id]
		return (math.exp(result), y)

	def get_probality(self, X):
		Pyxs = [(self.get_pxy(X, y)) for y in self._Y]
		Z = sum([prob for prob, y in Pyxs])
		return [(prob / Z, y) for prob, y in Pyxs]

	def get_EPx(self):
		self.EPx = [0.0 for i in range(self.n)]

		for i, X in enumerate(self._X):
			Pyxs = self.get_probality(X)

			for x in X:
				for Pyx, y in Pyxs:
					if self.fxy(x, y):
						id = self.xy2id[(x, y)]

						self.EPx[id] += Pyx * (1.0 / self.N)

	def fxy(self, x, y):
		return (x, y) in self.xy2id

	def train(self, X, Y):
		self.init_params(X, Y)
		self.w = [0.0 for i in range(self.n)]

		max_iter = 1000
		for times in range(max_iter):
			print('iterater times %d' % times)
			sigmas = []
			self.get_EPx()

			for i in range(self.n):
				sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
				sigmas.append(sigma)
			self.w = [self.w[i] + sigmas[i] for i in range(self.n)]

	def predict(self, testset):
		results = []
		for test in testset:
			result = self.get_probality(test)
			results.append(max(result, key=lambda x: x[0])[1])
		return results

def rebuild_features(features):
	'''
	将原feature的（a0,a1,a2,a3,a4,...）
	变成 (0_a0,1_a1,2_a2,3_a3,4_a4,...)形式
	'''
	new_features = []
	for feature in features:
		new_feature = []
		for i, f in enumerate(feature):
			new_feature.append(str(i) + '_' + str(f))
		new_features.append(new_feature)
	return new_features

def main():
	print('Start read data')

	time_1 = time.time()

	raw_data = pd.read_csv('../corpus/train_binary.csv', header=0)
	data = raw_data.values

	imgs = data[0::, 1::]
	labels = data[::, 0]

	# 选取 2/3 数据作为训练集， 1/3 数据作为测试集
	train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)

	train_features = rebuild_features(train_features)
	test_features = rebuild_features(test_features)

	time_2 = time.time()
	print('read data cost ', time_2 - time_1, ' second', '\n')

	print('Start training')
	met = MaxEnt()
	met.train(train_features, train_labels)

	time_3 = time.time()
	print('training cost ', time_3 - time_2, ' second', '\n')

	print('Start predicting')
	test_predict = met.predict(test_features)
	time_4 = time.time()
	print('predicting cost ', time_4 - time_3, ' second', '\n')

	score = accuracy_score(test_labels, test_predict)
	print("The accruacy socre is ", score)

if __name__ == '__main__':
	main()