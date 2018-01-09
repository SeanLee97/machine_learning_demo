# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def load_dataset(filename):
	data_mat = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			float_line = map(float, line.strip().split('\t'))
			data_mat.append(float_line)
	return data_mat

# 计算欧式距离
def dist_eclud(vec_a, vec_b):
	return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))

# 构建聚簇中心
def rand_center(dataset, k):
	n = np.shape(dataset)[1]
	centroids = np.mat(np.zeros((k, n)))
	for j in range(n):
		min_j = np.min(dataset[:,j])
		max_j = np.max(dataset[:,j])
		range_j = float(max_j-min_j)
		centroids[:,j] = min_j+range_j*np.random.rand(k, 1)
	return centroids
 
def KMeans(dataset, k, dist_means=dist_eclud, create_center=rand_center):
	m,_ = np.shape(dataset)
	cluster_assment = np.mat(np.zeros((m,2)))
	centroids = create_center(dataset, k)
	cluster_changed = True
	while cluster_changed:
		cluster_changed = False
		for i in range(m):
			min_dist = np.inf # 初始化为无穷大
			min_index = -1
			# 寻找最近质心
			for j in range(k):
				dist_ji = dist_means(centroids[j, :], dataset[i,:])
				if dist_ji < min_dist:
					min_dist = dist_ji
					min_index = j
			if cluster_assment[i,0] != min_index:
				cluster_changed = True
			cluster_assment[i,:] = min_index, min_dist**2
		for cent in range(k):
			pts_inclust = dataset[np.nonzero(cluster_assment[:,0].A ==cent)[0]]
			centroids[cent,:] = np.mean(pts_inclust, axis=0)
	return centroids, cluster_assment

# 二份k-均值
'''
克服k-均值算法收敛于局部最小值
'''
def bi_kmeans(dataset, k, dist_means=dist_eclud):
	m,_ = np.shape(dataset)
	cluster_assment = np.mat(np.zeros((m,2)))
	centroid0 = np.mean(dataset, axis=0).tolist()[0]
	cent_list = [centroid0]
	for j in range(m):
		cluster_assment[j,1] = dist_means(np.mat(centroid0), dataset[j,:])**2
	while len(cent_list) < k:
		lowest_SSE = np.inf
		for i in range(len(cent_list)):
			pts_incur_cluster = dataset[np.nonzero(cluster_assment[:,0].A==i)[0], :]
			centroid_mat, split_clust_ass = KMeans(pts_incur_cluster, 2, dist_means)
			sse_split = np.sum(split_clust_ass[:,1])
			sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:,0].A!=i)[0], 1])
			if sse_split+sse_not_split < lowest_SSE:
				best_cent_to_split = i
				best_new_cents = centroid_mat
				best_clust_ass = split_clust_ass.copy()
				lowest_SSE = sse_split+sse_not_split
		best_clust_ass[np.nonzero(best_clust_ass[:,0].A == 1)[0], 0] = len(cent_list)
		best_clust_ass[np.nonzero(best_clust_ass[:,0].A == 0)[0], 0] = best_cent_to_split
		cent_list[best_cent_to_split] = best_new_cents[0,:].tolist()[0]
		cent_list.append(best_new_cents[1,:].tolist()[0])
		cluster_assment[np.nonzero(cluster_assment[:,0].A==best_cent_to_split)[0],:] = best_clust_ass
	return np.mat(cent_list), cluster_assment
