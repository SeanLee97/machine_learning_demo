# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Sean
@description: load data
'''

def load_car():
	import pandas as pd
	car = pd.read_csv('corpus/car.csv', names=['buying','maint','doors','persons','lug_boot','safety', 'class'])
	return car.values.tolist(), list(car.columns)[:-1]

def load_dataset():
	"""
	创建数据集
	"""
	dataset = [['1000','youth', 'N', 'N', 'normal', 'disagree'],
				['2000','youth', 'N', 'N', 'good', 'disagree'],
				['7000','youth', 'Y', 'N', 'good', 'agree'],
				['7100','youth', 'Y', 'Y', 'normal', 'agree'],
				['3000','youth', 'N', 'N', 'normal', 'disagree'],
				['3500','mid-age', 'N', 'N', 'normal', 'disagree'],
				['3600','mid-age', 'N', 'N', 'good', 'disagree'],
				['8000','mid-age', 'Y', 'Y', 'good', 'agree'],
				['9000','mid-age', 'N', 'Y', 'very_good', 'agree'],
				['9200','mid-age', 'N', 'Y', 'very_good', 'agree'],
				['8600','elder', 'N', 'Y', 'very_good', 'agree'],
				['7800','elder', 'N', 'Y', 'good', 'agree'],
				['10000','elder', 'Y', 'N', 'good', 'agree'],
				['6500','elder', 'Y', 'N', 'very_good', 'agree'],
				['3000','elder', 'N', 'N', 'normal', 'disagree'],
			]
	labels = ['salary','age', 'job', 'house', 'loan']
	# 返回数据集和每个维度的名称
	return dataset, labels

def load_mini():
	dataset=[[1,1,'yes'],
		[1,1,'yes'],
		[1,0,'no'],
		[0,1,'no'],
		[0,1,'no']]
	labels = ['no surfaceing','flippers']
	return dataset, labels
