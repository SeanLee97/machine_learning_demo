# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

# 构建第一个候选集
def create_C1(dataset):
	C1 = []
	for transaction in dataset:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	#print(C1)
	return map(frozenset, C1)  # frozenset() 不可改变的集合

# 获取频繁项
# @param D 经过set映射后的dataset
# @param CK 候选集
# @param min_support 最小支持度
def scan_D(D, CK, min_support):
	ss_cnt = {}   # 字典
	for tid in D:
		for can in CK:
			if can.issubset(tid):
				if not can in ss_cnt:
					ss_cnt[can] = 1
				else:
					ss_cnt[can] += 1
	num_items = float(len(D))
	if num_items == 0:
		print("num items is empty")
		exit()
	ret_list = []
	support_data = {}
	for key in ss_cnt:
		support = ss_cnt[key]/num_items # 计算支持度
		if support >= min_support:
			ret_list.insert(0, key)
		support_data[key] = support
	return ret_list, support_data

# 从频繁集中生成候选集CK
def apriori_gen(LK, k):
	ret_list = []
	len_LK = len(LK)
	for i in range(len_LK):
		for j in range(i+1, len_LK):
			L1 = list(LK[i])[:k-2]
			L2 = list(LK[j])[:k-2]
			L1.sort()
			L2.sort()
			print(L1, '<>', L2)
			if L1 == L2:
				ret_list.append(LK[i] | LK[j]) #LK[i]|LK[j] 求并集
	return ret_list

# 实现apriori算法
def apriori(dataset, min_support = 0.5):
	C1 = create_C1(dataset)
	D = list(map(set, dataset))	
	L1, support_data = scan_D(D, C1, min_support)
	L = [L1]
	k = 2
	while len(L[k-2]) > 0:
		CK = apriori_gen(L[k-2], k)
		LK, supK = scan_D(D, CK, min_support)
		support_data.update(supK)
		L.append(LK)
		k += 1
	return L, support_data

# 生成关联规则
def generate_rules(L, support_data, min_conf = 0.7):
	big_rule_list = []
	for i in range(1, len(L)):
		# 只获取有两个或更多元素的集合
		for freq_set in L[i]:
			H1 = [frozenset([item]) for item in freq_set]
			if i>1:
				rules_from_conseq(freq_set, H1, support_data, big_rule_list, min_conf)
			else:
				calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
	return big_rule_list

# 计算可信度值
# brl big_rule_list
def calc_conf(freq_set, H, support_data, brl, min_conf=0.7):
	pruned_H = []
	for conseq in H:
		conf = support_data[freq_set]/support_data[freq_set-conseq]
		if conf > min_conf:
			print(freq_set-conseq, '-->', conseq, 'conf: ', conf)
			brl.append((freq_set-conseq, conseq, conf))
			pruned_H.append(conseq)
	return pruned_H

def rules_from_conseq(freq_set, H, support_data, brl, min_conf=0.7):
	m = len(H[0])
	if len(freq_set) > m+1:
		Hmp1 = apriori_gen(H, m+1)
		Hmp1 = calc_conf(freq_set, Hmp1, support_data, brl, min_conf)
		if len(Hmp1) > 1:
			rules_from_conseq(freq_set, Hmp1, support_data, brl, min_conf)


