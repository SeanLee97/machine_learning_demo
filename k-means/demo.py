# !/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from Kmeans import *


def dist_SLC(vecA, vecB):#Spherical Law of Cosines
	a = np.sin(vecA[0,1]* np.pi/180) * np.sin(vecB[0,1]* np.pi/180)
	b = np.cos(vecA[0,1]* np.pi/180) * np.cos(vecB[0,1]* np.pi/180) * np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
	return np.arccos(a + b)*6371.0 #pi is imported with numpy

def cluster_clubs(num_clust=5):
	dat_list = []
	for line in open('places.txt').readlines():
		line_arr = line.split('\t')
		dat_list.append([float(line_arr[4]), float(line_arr[3])])
	dat_mat = np.mat(dat_list)
	my_centroids, clust_assing = bi_kmeans(dat_mat, num_clust, dist_means=dist_SLC)
	fig = plt.figure()
	rect=[0.1,0.1,0.8,0.8]
	scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
	axprops = dict(xticks=[], yticks=[])
	ax0=fig.add_axes(rect, label='ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1=fig.add_axes(rect, label='ax1', frameon=False)
	for i in range(num_clust):
		ptsInCurrCluster = dat_mat[np.nonzero(clust_assing[:,0].A==i)[0],:]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
	ax1.scatter(my_centroids[:,0].flatten().A[0], my_centroids[:,1].flatten().A[0], marker='+', s=300)
	plt.show()

cluster_clubs()