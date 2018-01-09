# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from NBayes import NBayes

def load_data():
	dataset = [
		['my', 'dog', 'has','flea','problems','help','please', 'dog', 'is', 'cute'],
		['maybe','not','take','him','to','dog','park','stupid'],
		['my','dalmation','is','so','cute','I','love','him'],
		['stop','posting','stupid','worthless','garbage'],
		['mr','licks','ate','my','steak','how','to','stop','him'],
		['quit','buying','worthless','dog','food','stupid']
	]
	category = [0,1,0,1,0,1]
	return dataset, category

def create_vocab_list(dataset):
	vocab_set = set([])
	for doc in dataset:
		vocab_set = vocab_set | set(doc)
	return list(vocab_set)

# 词集向量
def set_of_word2vec(vocab_list, input_set):
	result_vec = [0] * len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			result_vec[vocab_list.index(word)] = 1
	return result_vec

# 词袋向量
def bag_of_word2vec(vocab_list, input_set):
	result_vec = [0] * len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			result_vec[vocab_list.index(word)] += 1
	return result_vec

def main():
	dataset, category = load_data()
	vocab_list = create_vocab_list(dataset)
	# print(vocab_list)
	matrix = []
	for doc in dataset:
		matrix.append(bag_of_word2vec(vocab_list, doc))
	print('bag bow>')
	print(matrix)
	matrix1 = []
	for doc in dataset:
		matrix1.append(set_of_word2vec(vocab_list, doc))
	print('set bow>')
	print(matrix1)
	nb = NBayes(matrix, category)
	'''
	p0, p1, pa = nb.train()
	print('p0 > ', p0)
	print('p1 > ', p1)
	print('pa > ', pa)
	'''
	test_doc = ['love', 'my', 'dalmaton']
	test_matrix = np.array(bag_of_word2vec(vocab_list, test_doc))
	print(test_doc, ' classified as > ', nb.classifier(test_matrix))
	test_doc = ['you', 'are', 'stupid']
	test_matrix = np.array(bag_of_word2vec(vocab_list, test_doc))
	print(test_doc, ' classified as > ', nb.classifier(test_matrix))

if __name__ == '__main__':
	main()
		
