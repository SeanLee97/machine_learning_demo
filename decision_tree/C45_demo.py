# !/usr/bin/env python
# -*- coding: utf-8 -*-

from data import load_car, load_dataset, load_mini
from decision.C45 import DecisionTree

dataset, attribute = load_mini()
print(dataset)
#print(attribute)
tree = DecisionTree(dataset, attribute)
tree.drawtree()
