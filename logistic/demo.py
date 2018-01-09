# !/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt 

def load_dataset():
	data_mat = []
	label_mat = []
	with open('data.txt', 'r') as f:
		for line in f.readlines():
			line_arr = line.strip().split('\t')
			data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
			label_mat.append(float(line_arr[2]))
	return data_mat, label_mat

# sigmoid函数
def sigmoid(x):
	return 1.0/(1+np.exp(-x))

# 求解最佳回归系数
# 梯度上升法（求最大值）
# 梯度下降法（求最小值）
def grad_ascent(data_mat, label_mat):
	data_matrix = np.mat(data_mat) # 数组转矩阵
	label_matrix = np.mat(label_mat).transpose() # 数组转矩阵且将矩阵转置，这样data_matrix就一行对应一个类别了
	m, n = np.shape(data_matrix)
	alpha = 0.001 # 初始化向目标移动的步长
	max_cycles = 500 # 迭代次数
	weights = np.ones((n,1)) # 初始化回归系数
	# print(weights)
	for i in range(0, max_cycles):
		# 以下方法是求函数梯度的方法至于具体的原理可以参见
		# http://blog.sina.com.cn/s/blog_61f1db170101k1wr.html
		h = sigmoid(data_matrix*weights)
		error = label_matrix - h # 计算误差
		weights = weights + alpha * data_matrix.transpose() * error
	return weights

# 随机梯度上升
def SGA_0(data_matrix, label_matrix):
	data_matrix = np.array(data_matrix)
	m, n = np.shape(data_matrix)
	alpha = 0.01
	weights = np.ones(n)
	for i in range(0, m):
		h = sigmoid(np.sum(data_matrix[i] * weights))
		error = label_matrix[i] - h
		weights = weights + alpha * error * data_matrix[i]
	return weights

# 优化随机梯度上升
def SGA(data_matrix, label_matrix, num_iter = 150):
	m, n = np.shape(data_matrix)
	weights = np.ones(n)
	for i in range(0, num_iter):
		data_index = list(range(m))
		for j in range(0, m):
			alpha = 4 / (1.0+j+i)+0.01 # alpha在每次 迭代的时候都会调整，会缓解数据波动和高频波动，另外alpha会随着迭代次数不断减小，但永远不会减小到0
			rand_index = int(random.uniform(0, len(data_index)))
			h = sigmoid(np.sum(data_matrix[rand_index]*weights))
			error = label_matrix[rand_index] - h
			weights = weights + alpha * error * data_matrix[rand_index]
			del(data_index[rand_index])
	return weights

# 画图
def best_fit(weight, data_matrix, label_matrix):
	weights = weight   # 矩阵转列表
	data_arr = np.array(data_matrix) # 矩阵转数组
	n,_ = np.shape(data_matrix)
	xcord1, ycord1 = [], []
	xcord2, ycord2 = [], []
	
	for i in range(n):
		if int(label_mat[i]) == 1:
			xcord1.append(data_arr[i, 1])
			ycord1.append(data_arr[i, 2])
		else:
			xcord2.append(data_arr[i, 1])
			ycord2.append(data_arr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('x1')
	plt.xlabel('x2')
	plt.show()

if __name__ == '__main__':
	data_mat, label_mat = load_dataset()
	'''
	weight = grad_ascent(data_mat, label_mat)
	best_fit(weight.getA(), data_mat, label_mat)
	'''
	'''
	weight = SGA_0(data_mat, label_mat)
	best_fit(weight, data_mat, label_mat)
	'''
	weight = SGA(np.array(data_mat), label_mat)
	best_fit(weight, data_mat, label_mat)
	
