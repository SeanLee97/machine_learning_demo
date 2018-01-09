# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import operator
import os

class KNN(object):
	"""docstring for KNN"""
	# inX input value
	# dataset
	# labels
	# k 选择最近邻居的数量
	def __init__(self, in_x, dataset, labels, k):
		super(KNN, self).__init__()
		self.in_x = in_x
		self.dataset = dataset
		self.labels = labels
		self.k = k

	def classifier(self):
		# 采用欧式距离来计算距离
		dataset_size = self.dataset.shape[0]

		# calculate distance
		diff = np.tile(self.in_x, (dataset_size, 1)) - self.dataset
		sq_diff = diff**2
		#print("sq_diff > ", sq_diff)
		sq_distance = sq_diff.sum(axis=1) # x 轴求和
		distance = sq_distance**0.5

		# 选择距离最小的K个点
		sorted_distance_indicies = distance.argsort()
		class_count = {}
		for i in range(self.k):
			vote_label = self.labels[sorted_distance_indicies[i]]
			class_count[vote_label] = class_count.get(vote_label, 0) + 1
		sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
		return sorted_class_count[0][0]

def loadDataSet():  
    dataSetDir = '../corpus/'  
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set  
    numSamples = len(trainingFileList)  
  
    train_x = np.zeros((numSamples, 1024))  
    train_y = []  
    for i in range(numSamples):  
        filename = trainingFileList[i]  
  
        # get train_x  
        train_x[i, :] = img2vec(dataSetDir + 'trainingDigits/%s' % filename)   
  
        label = int(filename.split('_')[0]) # return 1  
        train_y.append(label)  
  
    testingFileList = os.listdir(dataSetDir + 'testDigits') # load the testing set  
    numSamples = len(testingFileList)  
    test_x = np.zeros((numSamples, 1024))  
    test_y = []  
    for i in range(numSamples):  
        filename = testingFileList[i]  
  
        # get train_x  
        test_x[i, :] = img2vec(dataSetDir + 'testDigits/%s' % filename)   
  
        label = int(filename.split('_')[0]) # return 1  
        test_y.append(label)  
  
    return train_x, train_y, test_x, test_y  
  
def testHandWritingClass():  
    train_x, train_y, test_x, test_y = loadDataSet()  
  
    numTestSamples = test_x.shape[0]  
    matchCount = 0  
    for i in range(numTestSamples):  
        predict = KNN(test_x[i], train_x, train_y, 3).classifier()
        print(predict)
        if predict == test_y[i]:  
            matchCount += 1  
    accuracy = float(matchCount) / numTestSamples  
  
    ## show the result  
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))

def img2vec(filename):
	rows, cols = 32, 32
	imgVec = np.zeros((1, rows * cols)) # shape (1, 1024)
	with open(filename) as f:
		for row in range(rows):
			line = f.readline()
			for col in range(cols):
				imgVec[0, row * 32 + col] = int(line[col])

	return imgVec


def main():
	testHandWritingClass()
if __name__ == '__main__':
	main()