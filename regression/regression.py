# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_dataset(filename):
	data_mat = []
	label_mat = []
	with open(filename, 'r') as f:
		data = f.readlines()
		line_0 = data[0]
		#print(f.readlines())
		num_feat = len(line_0.split('\t')) - 1
		for line in data:
			line_arr = []
			cur_line = line.strip().split('\t')
			for i in range(num_feat):
				line_arr.append(float(cur_line[i]))
			data_mat.append(line_arr)
			label_mat.append(float(cur_line[-1]))
	return data_mat, label_mat
'''
求最佳拟合直线
'''
def stand_regres(x_arr, y_arr):
	x_mat = np.mat(x_arr)
	y_mat = np.mat(y_arr).T
	xTx = x_mat.T*x_mat
	if np.linalg.det(xTx) == 0.0: # 判断是否为正定矩阵，从而判断是否可逆，若为0则不可逆
		print('Sorry, xTx is not inversable.')
		exit()
	w_best = xTx.I*(x_mat.T*y_mat)
	#w_best = np.linalg.solve(xTx.I, x_mat.T*y_mat)
	return w_best

# 局部加权线性回归(LWLR)
def LWLR(test_point, x_arr, y_arr, k=1.0):
	x_mat = np.mat(x_arr)
	y_mat = np.mat(y_arr).T
	m = np.shape(x_mat)[0]
	weights = np.mat(np.eye(m))
	for j in range(m):
		diff_mat = test_point - x_mat[j,:]
		weights[j,j] = np.exp(diff_mat*diff_mat.T/-2*k**2)
	xTx = x_mat.T*(weights*x_mat)
	if np.linalg.det(xTx) == 0.0:
		print('Sorry, xTx is not inversable.')
		exit()
	w_best = xTx.I*(x_mat.T*(weights*y_mat))
	return test_point * w_best		
def LWLR_test(test_arr, x_arr, y_arr, k=1.0):
	m = np.shape(test_arr)[0]
	y_hat = np.zeros(m)
	for i in range(m):
		y_hat[i] = LWLR(test_arr[i], x_arr, y_arr, k)
	return y_hat
	
# 岭回归
def ridge_regres(x_mat, y_mat, lam=0.2):
	xTx = x_mat.T*x_mat
	_, n = np.shape(x_mat)
	I = np.eye(n)
	denom = xTx+lam*I
	if np.linalg.det(denom) == 0.0:
		print('Sorry, xTx is not inversable.')
		exit()
	w_best = denom.I*(x_mat.T*y_mat)
	return w_best
def ridge_test(x_arr, y_arr):
	x_mat = np.mat(x_arr)
	y_mat = np.mat(y_arr).T
	y_mean = np.mean(y_mat, axis=0)  # 求平均值
	# 以下进行数据规整
	y_mat = y_mat - y_mean
	x_mean = np.mean(x_mat, axis=0)
	x_var = np.var(x_mat, axis=0)  # 求x_mat的方差
	x_mat = (x_mat - x_mean)/x_var
	
	num_test_pts = 30
	w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
	for i in range(num_test_pts):
		ws = ridge_regres(x_mat, y_mat, np.exp(i-10))
		w_mat[i,:] = ws.T
	return w_mat

'''
# 前向逐步线性回归
# 贪心算法，每一步尽可能减少误差
'''
def regularize(x_mat):#regularize by columns
	in_mat = x_mat.copy()
	in_means = np.mean(in_mat, axis=0)   #calc mean then subtract it off
	in_var = np.var(in_mat, axis=0)      #calc variance of Xi then divide by it
	in_mat = (in_mat - in_means)/in_var
	return in_mat
def rss_error(y_arr, y_hat_arr):
	return ((y_arr-y_hat_arr)**2).sum()

def stage_wise(x_arr, y_arr, eps=0.01, num_iter=100):
	x_mat = np.mat(x_arr)
	y_mat = np.mat(y_arr).T
	y_mean = np.mean(y_mat, axis=0)
	y_mat = y_mat - y_mean
	x_mat = regularize(x_mat)
	m, n = np.shape(x_mat)
	return_mat = np.zeros((num_iter, n))
	w_best = np.zeros((n, 1))
	w_best_test = w_best.copy()
	w_best_max = w_best.copy()
	for i in range(num_iter):
		lowest_error = np.inf
		for j in range(n):
			for sign in [-1, 1]:
				w_best_test = w_best.copy()
				w_best_test[j] += eps*sign
				y_test = x_mat*w_best_test
				rss_e = rss_error(y_mat.A, y_test.A)
				if rss_e < lowest_error:
					lowest_error = rss_e
					w_best_max = w_best_test
		w_best = w_best_max.copy()
		return_mat[i,:] = w_best.T
	return return_mat

def main():
	'''
	x_arr, y_arr = load_dataset('ex0.txt')
	# 线性回归
	#print(x_arr[:2])
	ws = stand_regres(x_arr, y_arr) # ws存放回归系数
	#print(ws)
	x_mat = np.mat(x_arr)
	y_mat = np.mat(y_arr)
	fig = plt.figure('regression')
	ax = fig.add_subplot(111)
	ax.scatter(x_mat[:,1].flatten().A[0], y_mat.T[:,0].flatten().A[0])
	x_copy = x_mat.copy()
	x_copy.sort(0)
	y_hat = x_copy*ws
	ax.plot(x_copy[:,1], y_hat, color='r')  # 画出回归直线
	plt.show()
	'''
	# 单点估计
	#print(LWLR(x_arr[0], x_arr, y_arr, 1.0))
	
	'''
	# 局部加权线性回归
	x_arr, y_arr = load_dataset('ex0.txt')
	y_hat = LWLR_test(x_arr, x_arr, y_arr, 0.003)
	#print(y_hat)
	x_mat = np.mat(x_arr)
	srt_ind = x_mat[:, 1].argsort(0) # 对x_arr排序
	x_sort = x_mat[srt_ind][:, 0, :] 
	fig = plt.figure('LWLR')
	ax = fig.add_subplot(111)
	ax.plot(x_sort[:,1], y_hat[srt_ind])
	ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0], s=2, c='red')
	plt.show()
	'''
	
	'''
	# 岭回归
	x_arr, y_arr = load_dataset('abalone.txt')
	ridge_weights = ridge_test(x_arr, y_arr)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridge_weights)
	plt.show()
	'''
	
	'''
	# lasso 缩减算法省略
	'''

	'''
	# 前向逐步线性回归
	'''	
	x_arr, y_arr = load_dataset('abalone.txt')
	print(stage_wise(x_arr, y_arr, 0.01, 200))
	
if __name__ == '__main__':
	main()
