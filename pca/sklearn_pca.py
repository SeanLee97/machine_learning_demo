# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
data = iris.data

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

print('特征值占的比例', pca.explained_variance_ratio_)
print('降维后占比', sum(pca.explained_variance_ratio_))

print(pca_data)
