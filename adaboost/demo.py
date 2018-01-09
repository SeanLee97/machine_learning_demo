# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from AdaBoost import *
'''
构造数据集
'''
def load_dataset():
	data_mat = np.matrix([
		[1.0, 2.1],
		[2. , 1.1],
		[1.3, 1. ],
		[1. , 1. ],
		[2. , 1. ]
	])
	labels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return data_mat, labels

if __name__ == '__main__':
	data_mat, labels = load_dataset()
	classifier_arr = ada_boost_train_ds(data_mat, labels, 30)
	print("[0,0]: ", ada_classify([0, 0], classifier_arr))
	print("[5,5], [0,0]: ", ada_classify([[5,5],[0,0]], classifier_arr))	
