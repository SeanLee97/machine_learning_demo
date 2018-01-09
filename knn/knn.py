# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import operator

class KNN(object):
	"""docstring for KNN"""
	# inX input value
	# dataset
	# labels
	# k 选择最近邻居的数量
	def __init__(self, in_x, dataset, labels, k):
		super(KNN, self).__init__()
		self.in_x = in_x
		self.dataset = dataset
		self.labels = labels
		self.k = k

	def classifier(self):
		# 采用欧式距离来计算距离
		dataset_size = self.dataset.shape[0]

		# calculate distance
		diff = np.tile(self.in_x, (dataset_size, 1)) - self.dataset
		sq_diff = diff**2
		#print("sq_diff > ", sq_diff)
		sq_distance = sq_diff.sum(axis=1) # x 轴求和
		distance = sq_distance**0.5

		# 选择距离最小的K个点
		sorted_distance_indicies = distance.argsort()
		class_count = {}
		for i in range(self.k):
			vote_label = self.labels[sorted_distance_indicies[i]]
			class_count[vote_label] = class_count.get(vote_label, 0) + 1
		sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
		return sorted_class_count[0][0]

def set_dataset():
	group = np.array([[1, 1], [1, 1.2], [0, 0], [0, 0.2], [3, 0.5], [3.3, 0.9]])
	labels = ['A', 'A', 'B', 'B', 'C', 'C']
	return group, labels

def main():
	dataset, labels = set_dataset()
	pred = KNN([4,1], dataset, labels, 3).classifier()
	print(pred)
if __name__ == '__main__':
	main()