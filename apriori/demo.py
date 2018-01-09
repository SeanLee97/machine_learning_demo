# !/usr/bin/env python
# -*- coding:utf-8 -*-

from Apriori import *

def load_dataset():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def main():
	dataset = load_dataset()
	L, support_data = apriori(dataset, 0.2)	
	rules = generate_rules(L, support_data, min_conf = 0.7)
	print(rules)
if __name__ == '__main__':
	main()
