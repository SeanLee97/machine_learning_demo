# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import preprocessing # 预处理，数据标准化
import numpy as np

a = np.array([
	[10, 2.7, 3.6],
	[-100, 5, -2],
	[120, 20, 40]
], dtype=np.float64)

print(a)

# 计算公式 (X-X_mean)/X_std 去除均值和方差进行缩放
# sklearn.preprocessing.scale(X, axis=0, with_mean=True,with_std=True,copy=True)
# http://blog.csdn.net/dream_angel_z/article/details/49406573
print('标准化> ', preprocessing.scale(a))

# 将特征取值压缩到指定区间
min_max_scaler = preprocessing.MinMaxScaler() # 默认压缩到0-1范围
X_minMax = min_max_scaler.fit_transform(a)
print("[0, 1]范围> ",X_minMax) 

# StandardScaler 计算训练集的平均值和标准差
scaler = preprocessing.StandardScaler().fit(a)
print("均值> ", scaler.mean_)	# 均值
#print("标准差> ", scaler.std_)  # 标准差
print("标准差> ", scaler.scale_) # 新版标准差求法
print("标准化> ", scaler.transform(a)) # 效果同上面的标准化
print("输入数据标准化> ", scaler.transform([[1, 1, 0]]))

# 绝对值最大标准化
# 与上述标准化方法相似，但是它通过除以最大值将训练集缩放至[-1,1]。这意味着数据已经以０为中心或者是含有非常非常多０的稀疏数据
max_abs_scaler = preprocessing.MaxAbsScaler()
x_train_maxabs = max_abs_scaler.fit_transform(a)
x_test = np.array([[-3,-1,4]])
x_test_maxabs = max_abs_scaler.transform(x_test)
print('绝对值最大标准化> ', x_test_maxabs) 
