# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PCA Principal Component Analysis 主成分分析
可对高维数据进行降维，去除噪声，从而发现数据中一些固有模式
PCA把原先N个特征用数目更少的M个特征代替，新特征是旧特征的线性组合，这些线性组合最大化了样本方差，尽可能使新的M个特征互不相干。
PCA的实现方式一般有两种：
1. SVD （sklearn基于SVD方式实现）
2. 特征值分解

本例子基于特征值分解实现
'''

import numpy as np
from sklearn.datasets import load_iris

# 降至2维
n_components = 2

iris = load_iris()

data = iris.data
print('Data Shape：', data.shape)

# 1. 对每个特征（每一列）进行零均值化，即每个数据减去平均值
mean_val = np.mean(data, axis=0)
mean_data = data - mean_val
#print(mean_data)

# 2. 求协方差方阵
cov_mat = np.cov(mean_data, rowvar=False) # rowvar是否行作为特征，在此列为特征，故rowvar=False
#print(cov_mat)
print('Cov Shape:', cov_mat.shape)

# 3. 求特征值和特征向量
# A是n阶方阵，如果存在数λ和n维非零列向量X使得AX = λX成立，则λ成为A的特征值，列向量X是λ对应的特征向量
eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
#print('特征值：', eig_vals)
#print('特征向量：', eig_vects)

# 4. 对特征值从大到小排序
sorted_index = np.argsort(-eig_vals) # -倒序, 注意获得是排序后的索引值index
print('sorted：', sorted_index)

# 5. 降维
topn_index = sorted_index[:n_components]
topn_vects = eig_vects[:, topn_index]
print('Topn_vects Shape：', topn_vects.shape)
pca_data = data * topn_vects # 投影到低维空间
print('降维后的数据', pca_data)
