# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class NBayes(object):
	"""docstring for NBayes"""
	def __init__(self, matrix, category):
		super(NBayes, self).__init__()
		self._matrix = matrix
		self._category = category

	def train(self):
		'''
		print('matrix> ', self._matrix)
		print("matrix[0] > ", self._matrix[0])
		print("words_num > ", len(self._matrix[0]))
		'''
		docs_num = len(self._matrix)
		words_num = len(self._matrix[0])
		# 侮辱性的文档所占总文档的概率
		p_abuse = sum(self._category)/float(docs_num)
		# 非侮辱性的词个数
		p_0 = np.zeros(words_num)
		# 侮辱性的词个数
		p_1 = np.zeros(words_num)
		# p0文档的总词数+1
		p_0_denom = 0.0
		# p1文档的总词数+1
		p_1_denom = 0.0
		for i in range(docs_num):
			if self._category[i] == 1:
				p_1 += self._matrix[i]   # numpy向量相加
				p_1_denom += sum(self._matrix[i])
			else:
				p_0 += self._matrix[i]
				p_0_denom += sum(self._matrix[i])

		p_1_vect = p_1/p_1_denom # p(Wi|C1)
		p_0_vect = p_0/p_0_denom # p(Wi|C0)
		return p_0_vect, p_1_vect, p_abuse

	def classifier(self, test_matrix):
		p_0_vec, p_1_vec, pa = self.train()
		p0 = sum(test_matrix * p_0_vec) + np.log(1 - pa)
		p1 = sum(test_matrix * p_1_vec) + np.log(pa)
		if p1 > p0:
			return 1
		else:
			return 0


