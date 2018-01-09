# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
'''
单层决策树生成函数
通过阙值比较对数据进行分类
'''
def stump_classify(data_matrix, dim, thresh_val, thresh_insq):
	ret_arr = np.ones((np.shape(data_matrix)[0], 1))
	if thresh_insq == 'lt':
		ret_arr[data_matrix[:, dim] <= thresh_val] = -1.0
	else:
		ret_arr[data_matrix[:, dim] > thresh_val] = -1.0
	return ret_arr

'''
找到数据集上最佳的单层决策树
'''
def build_stump(data_arr, labels, D):
	data_matrix = np.mat(data_arr)
	label_mat = np.mat(labels).transpose() # 构造label矩阵并求其转置
	# label_mat = np.mat(labels).T # 求矩阵转置的另一种方法
	m, n = np.shape(data_matrix)
	num_steps = 10.0   # 用于在特征的所有可能值上进行便利
	best_stump = {}	   # 字典用于存储给定权重向量0时所得到的最佳单层决策树的相关信息
	best_class_ent = np.mat(np.zeros((m, 1)))
	min_error = np.inf    # 将min_error 初始无穷大
	for i in range(n):
		range_min = data_matrix[:, i].min()
		range_max = data_matrix[:, i].max()
		step_size = (range_max-range_min) / num_steps
		for j in range(-1, int(num_steps) + 1):
			for inequal in ['lt', 'gt']:
				thresh_val = (range_min + float(j) * step_size)
				predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
				err_arr = np.mat(np.ones((m, 1)))
				err_arr[predicted_vals==label_mat] = 0
				weighted_error = D.T * err_arr   # 计算加权错误概率
				if weighted_error < min_error:
					min_error = weighted_error
					best_class_ent = predicted_vals.copy()
					best_stump['dim'] = i
					best_stump['thresh'] = thresh_val
					best_stump['ineq'] = inequal
	return best_stump, min_error, best_class_ent

'''
基于单层决策树的AdaBoost训练过程
@param num_iter 迭代次数，默认40
'''
def ada_boost_train_ds(data_arr, labels, num_iter=40):
	weak_class_arr = []
	m = np.shape(data_arr)[0]
	D = np.mat(np.ones((m, 1))/m)
	agg_class_est = np.mat(np.zeros((m, 1)))
	# 开始迭代
	for i in range(num_iter):
		best_stump, error, class_est = build_stump(data_arr, labels, D)
		#print("D: ", D.T)
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) # 确保不会溢出
		best_stump['alpha'] = alpha
		weak_class_arr.append(best_stump)
		expon = np.multiply(-1*alpha*np.mat(labels).T, class_est)
		D = np.multiply(D, np.exp(expon))
		D = D/D.sum()
		agg_class_est += alpha*class_est
		agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(labels).T, np.ones((m, 1)))
		error_rate = agg_errors.sum()/m
		if error_rate == 0.0:
			break
	return weak_class_arr

'''
AdaBoost分类函数
'''
def ada_classify(data_to_class, classifier_arr):
	data_matrix = np.mat(data_to_class)
	m = np.shape(data_matrix)[0]
	agg_class_est = np.mat(np.zeros((m, 1)))
	for i in range(len(classifier_arr)):
		class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])
		agg_class_est += classifier_arr[i]['alpha']*class_est
	return np.sign(agg_class_est)
