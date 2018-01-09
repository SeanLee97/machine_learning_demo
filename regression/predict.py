# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
线性回归实例
预测鲍鱼年龄
'''
from regression import *

def rss_error(y_arr, y_hat_arr):
	return ((y_arr - y_hat_arr)**2).sum()

def main():
	ab_x, ab_y = load_dataset('abalone.txt')
	'''
	y_hat01 = LWLR_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
	y_hat1 = LWLR_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1.0)
	y_hat10 = LWLR_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)
	print('y_hat01', rss_error(ab_y[0:99], y_hat01.T))
	print('y_hat1', rss_error(ab_y[0:99], y_hat1.T))
	print('y_hat10', rss_error(ab_y[0:99], y_hat10.T))
	'''
	# 新样本
	y_hat01 = LWLR_test(ab_x[100:199], ab_x[100:199], ab_y[100:199], 0.1)
	y_hat1 = LWLR_test(ab_x[100:199], ab_x[100:199], ab_y[100:199], 1)
	y_hat10 = LWLR_test(ab_x[100:199], ab_x[100:199], ab_y[100:199], 10)
	print('---新样本---')
	print('y_hat01', rss_error(ab_y[100:199], y_hat01.T))
	print('y_hat1', rss_error(ab_y[100:199], y_hat1.T))
	print('y_hat10', rss_error(ab_y[100:199], y_hat10.T))
if __name__ == '__main__':
	main()
