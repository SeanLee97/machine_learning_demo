# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

x = np.array([[1, 2], [3, 4], [5, 6], [7,8]])
#print(x)

print(np.mean(x, axis=0, keepdims=True)) # [[4, 5]] axis=0对列进行求平均值，并保持维度不变
print(np.mean(x, axis=0, keepdims=False))# [4, 5] axis=0对行进行求平均值，维度变了

print(np.mean(x, axis=1, keepdims=True)) #[[1.5],[3.5],[5.5][7.5]] 对行进行求均值，并保持维度不变
print(np.mean(x, axis=1, keepdims=False))#[1.5 3.5 5.5 7.5]
