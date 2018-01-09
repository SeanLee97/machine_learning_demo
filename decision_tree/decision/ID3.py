# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: sean
@description: 基于ID3(Iterative Dichotomiser 3)实现决策树
ID3是以信息增益为准则来划分属性的，参考周志华《机器学习》编写
'''

import math

class DecisionTree(object):

	def __init__(self, D, A):
		self.tree = self.build_tree(D, A)

	# 创建决策树
	def build_tree(self, D, A):
		# D中样本是否全属于同一类别C
		C = [d[-1] for d in D]
		if len(C) > 0 and C.count(C[0]) == len(C):
			return C[0]
		# A为空集或者D中样本在A上取值相同
		if len(A) == 0 or len(set([''.join(map(str, d)) for d in D[:-1]]))==1:
			return self.most_example_C(C)
		best_dim = self.choose_best_split_attr(D, A)		
		best_attr = A[best_dim]	
		tree = {best_attr: {}}
		# 删属性
		del(A[best_dim])
		values = [d[best_dim] for d in D]
		values_set = set(values)
		for value in values_set:
			newD = self.subD(D, best_dim, value)
			newA = list(A)
			tree[best_attr][value] = self.build_tree(newD, newA)
		return tree

	def subD(self, D, dim, val):
		newD = []
		for d in D:
			if d[dim] == val:
				newd = d[:dim]
				newd.extend(d[dim+1:])
				newD.append(newd)
		return newD

	def drawtree(self):		
		from drawtree import draw
		draw(self.tree)

	# 计算信息熵
	def info_entropy(self, attrs_set, attrs):
		ent = 0
		for attr in attrs_set:
			proba = float(attrs.count(attr))/len(attrs)
			ent += proba*math.log(proba, 2)
		return -ent 	

	# 计算信息增益
	def info_entropy_gain(self, D, ent, dim):
		attrs = [d[dim] for d in D] # a*^v
		attrs_set = set(attrs)
		curent = self.info_entropy(attrs_set, attrs)
		gain = ent - curent
		return gain

	# 确定最优属性划分
	def choose_best_split_attr(self, D, A):
		dims = len(D[0])-1
		ent = self.info_entropy(set(A), A)
		best_gain = 0.0
		best_dim = -1
		for dim in range(dims):
			gain = self.info_entropy_gain(D, ent, dim)
			if gain > best_gain:
				best_gain = gain
				best_dim = dim
		return best_dim
			

	# 返回D中样本数最多的类
	def most_example_C(self, C):
		C_set = set(C)
		nums = []
		for c in C_set:
			nums.append((c, C.count(c)))
		nums_sorted = sorted(nums, key=lambda x: x[1], reverse=True)
		return nums_sorted[0][0]
