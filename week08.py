#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20160101(yyyymmdd)
# Updated on: 20170106(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)

# Built-in modules
import math
import random
import sys

# More popular package(s)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Less popular package(s)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Custom module/package(s)
from customModule import *

def showPlot_scores(scores):
	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
	plt.xlabel('gamma')
	plt.ylabel('C')
	plt.colorbar()
	plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
	plt.yticks(np.arange(len(C_range)), C_range)
	plt.title('Validation accuracy')
	plt.show()

class DataSet():
	def __init__(self):
		# Sequence of Xs
		# C1, red
		seqXC1sub1_20x2 = makeTranspose(genRandomVecs(20, 0.1**0.5, [0, 1]))
		seqXC1sub2_20x2 = makeTranspose(genRandomVecs(20, 0.1**0.5, [1, 0]))
		seqXC1_40x2 = seqXC1sub1_20x2 + seqXC1sub2_20x2

		# C2, blue
		seqXC2sub1_20x2 = makeTranspose(genRandomVecs(20, 0.1**0.5, [0, 0]))
		seqXC2sub2_20x2 = makeTranspose(genRandomVecs(20, 0.1**0.5, [1, 1]))
		seqXC2_40x2 = seqXC2sub1_20x2 + seqXC2sub2_20x2
		seqXC2Tp_2x40 = makeTranspose(seqXC2_40x2)
		self.seqX = seqXC1_40x2 + seqXC2_40x2 # 80x2

		# Sequence of Ys
		seqYC1_40x1 = [1,]*40
		seqYC2_40x1 = [-1,]*40
		self.seqY = seqYC1_40x1 + seqYC2_40x1 # 80x1

setTraining = DataSet()
setTest = DataSet()

# H8.2
print("H8.2")
C_range = np.logspace(-6, 10, num="9", base=2.0)
gamma_range = np.logspace(-5, 9, num="8", base=2.0)
param_grid = dict(gamma=gamma_range, C=C_range)
# grid = GridSearchCV(SVC(), param_grid=param_grid)
# grid.fit(setTraining.seqX, setTraining.seqY)

clf = SVC()
clf.fit(setTraining.seqX, setTraining.seqY)

countRight = 0
countTotal = 0

for rec in zip(setTraining.seqX, setTraining.seqY):
	if clf.predict([rec[0]])[0] == rec[1]:
		countRight += 1
	countTotal += 1
score = countRight/float(countTotal)
params = {'C':clf.get_params()['C'], 'gamma': clf.get_params()['gamma']}
print("The best parameters are %s with a score of %0.2f" %(params, score))

X = makeTranspose(setTraining.seqX)
Y = []
for y in setTraining.seqY:
	if y == 1:
		Y.append((1,0,0,1))
	elif y == -1:
		Y.append((0,0,1,1))

plt.clf()
plt.scatter(X[0], X[1], c=Y, zorder=10, cmap=plt.cm.Paired)

plt.axis('tight')
x_min = min(X[0])
x_max = max(X[0])
y_min = min(X[1])
y_max = max(X[1])
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.title("rbf")
plt.show()

# H8.3
C_range = np.logspace(-6, 10, num="9", base=2.0)
gamma_range = np.logspace(-5, 9, num="8", base=2.0)

# H8.3.A
print("\nH8.3.A")
param_grid = dict(C=C_range, gamma=gamma_range)
# cv = StratifiedShuffleSplit(test_size=0.25)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=4)
grid.fit(setTraining.seqX, setTraining.seqY)

print("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))	

showPlot_scores(scores)

# H8.3.B
print("\nH8.3.B")
# for CGamma in [(C, gamma) for C in C_range for gamma in gamma_range]:
scores = []
scoresLin = []
for C in C_range:
	elemScores = []
	for gamma in gamma_range:
		clf = SVC(C=C, gamma=gamma)
	 	clf.fit(setTraining.seqX, setTraining.seqY)
	
		countRight = 0
		countTotal = 0
		for rec in zip(setTraining.seqX, setTraining.seqY):
			if clf.predict([rec[0]])[0] == rec[1]:
				countRight += 1
			countTotal += 1
		elemScores.append(countRight/float(countTotal))
		scoresLin.append(countRight/float(countTotal))
		# print(clf.n_support_)
		if countRight/float(countTotal) == max(scoresLin):
			CBest = C
			gammaBest = gamma

	scores.append(elemScores)
showPlot_scores(scores)

paramsBest = {'C':CBest, 'gamma': gammaBest}
print("The best parameters are %s with a score of %0.2f" %(paramsBest, max(scoresLin)))

plt.clf()
plt.scatter(X[0], X[1], c=Y, zorder=10, cmap=plt.cm.Paired)

plt.axis('tight')
x_min = min(X[0])
x_max = max(X[0])
y_min = min(X[1])
y_max = max(X[1])
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.title("rbf")
plt.show()

# H8.3.C
print("\nH8.3.C")
C_range = np.logspace(-8, 8, num="9", base=2.0)
gamma_range = np.logspace(-7, 7, num="8", base=2.0)
param_grid = dict(C=C_range, gamma=gamma_range)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=4)
grid.fit(setTraining.seqX, setTraining.seqY)

print("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))	

showPlot_scores(scores)




