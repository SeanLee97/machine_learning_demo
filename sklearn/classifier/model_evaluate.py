# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ---outline---
# 1. pipeline
# 2. linear_curve
# 3. PCA
# 4. LogisticRegression
# 5. ROC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# load dataset
# 数据说明：数据集有569个样本，它的前两列为唯一的ID号和诊断结果 (M = malignant, B = benign) ，它的3->32列为实数值特征
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

'''
print(df.shape)
print(df.values[:2])
'''
x = df.iloc[:, 2:].values # data
y = df.iloc[:, 1].values  # label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 组装pipeline
#estimators = [('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1))]
# 为降维时过拟合严重，pca将31维压缩到了2维
estimators = [('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression())]
pipe_lr = Pipeline(estimators) # 组合模型
pipe_lr.fit(x_train, y_train)

#print(pipe_lr.score(x_test, y_test))

scores = cross_val_score(estimator=pipe_lr, X=x_test, y=y_test, cv=10, n_jobs=1)
print('test scores> ', scores)

# learning_curve
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=x_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.5, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right') # loc设置图例显示的位置
plt.ylim([0.8, 1.0])    # 设置y轴的范围
plt.show()

# 验证曲线
# 学习曲线时训练集数量与准确性间的函数。验证曲线时不同模型参数与准确性之间的函数
# param_range 存储SVM的惩罚系数C
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=x_train, y=y_train, param_name='clf__C', param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
# plt.fill_between() 填充区域
plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()

# 通过网格搜索选择合适的超参数搭配
# 调整SVM的C，kernel,gamma
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(x_train, y_train)
print('best_score> ', gs.best_score_)
print('best_params> ', gs.best_params_)

# 选择最佳模型为作为分类器
clf = gs.best_estimator_
clf.fit(x_train, y_train)
print('Test accuracy: %.3f' % clf.score(x_test, y_test))

# precision, recall, f1-score
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# F1 = (2*precision*recall)/(precision+recall)
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
#print(confmat)
fig, ax = plt.subplots()
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
	for j in range(confmat.shape[1]):
		ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

# ROC曲线
# ROC是一种选择分类模型基于ROC曲线，我们可以计算出描述分类模型性能的AUC（area under the curve）。在ROC曲线中，左下角的点所对应的是将所有样例判为反例的情况，而右上角的点对应的则是将所有样例判为正例的情况。
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
x_train2 = x_train[:, [4,14]]
cv = StratifiedKFlod(y_train, n_folds=3, random_state=1) # K折
fig = plt.figure()

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
	# 返回预测的每个类别（0， 1）的概率
	probas = pipe_lr.fit(x_train2[train], y_train[train]).predict_proba(x_train2[test])
	fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
	mean_tpr += interp(mean_fpr, fpr, tpr)
	mean_tpr[0] = 0.0
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, linewidth=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
# plot perfect performance line
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')
# 设置x，y坐标范围
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.show()
	
