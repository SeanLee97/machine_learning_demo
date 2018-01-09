# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# 加载数据集
def load_dataset(filename):
	data_mat = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			float_line = map(float, line.strip().split('\t'))
			data_mat.append(float_line)
	return data_mat

# 二元切分数据集合
# 需要了解numpy.nonzero()
def bin_split_dataset(dataset, feature, val):
	mat_gt = dataset[np.nonzero(dataset[:, feature]>val)[0], :]
	mat_lt = dataset[np.nonzero(dataset[:, feature]<=val)[0], :]
	return mat_gt, mat_lt

'''
mat = np.mat(np.eye(4))
print(mat)
a, b = bin_split_dataset(mat, 1, 0.5)
print(a)
print(b)
'''

# 求数据集的均值（即叶子节点的值）作为回归值
def reg_leaf(dataset):
	return np.mean(dataset[:, -1])

# 求数据集的总方差，用来衡量数据集的混合度
def reg_err(dataset):
	return np.var(dataset[:,-1]).shape(dataset)[0]

# ***选择最好的切分
'''
ops元组设置包括tolS,指切分前后误差的差的允许值，即两次误差的差必须大于这个限制值；另外一个是tolN，表示切分之后的两个子集的样本数必须大于这个值（相当于预剪枝）
'''
def choose_best_split(dataset, leaf_type=reg_leaf, err_type = reg_err, ops=(1, 4)):
	# err_limit 切分前后误差下降限制
	# num_limit 切分后子集的样本数目限制
	err_limit = ops[0]
	num_limit = ops[1]
	dataset = np.array(dataset)
	if len(set((dataset[:,-1].T).tolist()))==1:
		# 当所有y值都相同时停止
		return None, leaf_type(dataset)
	m, n = np.shape(dataset)
	s = reg_err(dataset)
	best_s = np.inf
	best_index = 0
	best_val = 0

	for featurn_index in range(n-1):
		for split_value in set(dataset[:,featurn_index]):
			greater, less = bin_split_dataset(dataset, feature_index, split_value)
			if np.shape(greater)[0] < num_limit or np.shape(less)[0]< num_limit:
				continue
			new_s = err_type(greater)+err_type(less)
			if new_s < best_s:
				best_index = feature_index
				best_value = split_value
				best_s = new_s
		if s-best_s < err_limit:
			# 划分前后的误差小于下降值则停止
			return None, leaf_type(dataset)
		greater, less = bin_split_dataset(dataset, best_index, best_value)
		if np.shape(greater)[0] < num_limit or np.shape(less)[0] < num_limit:
			# 切分后的两个数据集的样本数小于num_limit
			return None, leaf_type(dataset)
		return best_index, best_value

# 后剪枝
def is_tree():
	return type(obj).__name__=='dict'

def get_mean(tree):
	if is_tree(tree['right']):
		tree['right'] = get_mean(tree['right'])
	if is_tree(tree['left']):
		tree['left'] = get_mean(tree['left'])
	return (tree['right']+tree['left'])/2.0

# 剪枝函数
def prune(tree, test_data):
	if np.shape(test_data)[0] == 0:
		return get_mean(tree)
	if is_tree(tree['right']) or is_tree(tree['left']):
		lset, rset = bin_split_dataset(test_data, tree['sp_ind'], tree['sp_val'])
	if is_tree(tree['left']):
		tree['left']=prune(tree['left'], lset)
	if is_tree(tree['right']):
		tree['right']=prune(tree['right'], rset)
	if not is_tree(tree['right']) and not is_tree(tree['left']):
		lset, rset = bin_split_dataset(test_data, tree['sp_ind'], tree['sp_val'])
		# 没有合并的误差
		error_no_merge = np.sum(np.power(lset[:,-1]-tree['left'], 2))+np.sum(np.power(rset[:,-1]-tree['right'], 2))
		# 求合并后的误差
		tree_mean = (tree['left']+tree['right'])/2.0
		error_merge = np.sum(np.power(test_data[:,-1]-tree_mean, 2))
		# *比较前后误差，决定是否剪枝
		if error_merge < err_no_merge:
			return tree_mean
		else:
			return true
	return tree

# 模型树
'''
把节点设置为分段线性函数，分段线性是指模型由多个线性片段组成
'''
def linear_solve(dataset):
	data_mat = np.mat(dataset)
	m, n = np.shape(datamat)
	x = np.mat(np.ones((m,n)))
	y = np.mat(np.ones((m,1)))
	x[:,1:n] = data_mat[:,0:n-1]
	y = data_mat[:, -1]
	xTx = x.T*x
	if np.linalg.det(xTx) == 0.0:
		raise NameError('this matrix cannot do inverse!')
	ws = xTx.I*(x.T*y)
	return ws, x, y

'''
modelLeaf与modelErr这两个函数是用来生成叶节点的，不过生成的不是一个值而是一个线性模型；同理modelErr是用来计算误差的，这两个函数调用时，都会在里面调用linearModel函数，因为需要用划分的子数据集生成线性模型
'''  
def model_leaf(dataset):
	ws, x, y = linear_solve(dataset)
	return ws

def model_err(dataset):
	ws, x, y = linear_solve(dataset)
	y_hat = x*ws
	return np.sum(np.power(y-y_hat, 2))


