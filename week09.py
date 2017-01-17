#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20170106(yyyymmdd)
# Updated on: 20170109(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)

# Built-in modules
import math
import random
import sys

# More popular package(s)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from matplotlib.colors import Normalize

# Less popular package(s)
from sklearn.svm import NuSVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Custom module/package(s)
from customModule import *

# H9.2
class parseDataset():
	def __init__(self, src):
		self.src = src
		self.seqVecX = []
		self.seqY = []

		# Read data from the source
		self.readFile()

		# Render imported data
		self.seqVecX_Tp = makeTranspose(self.seqVecX)
	def readFile(self):
		with open(self.src, "r") as f:
			readline = f.readline()
			while readline:
				try:
					readline = readline.strip().split(",")
					
					vecX = []					
					for X in readline[:-1]:
						vecX.append(float(X))
					Y = float(readline[-1])
					
					self.seqY.append(Y)
					self.seqVecX.append(vecX)
					
				except ValueError:
					# print "ValueError: This row is the header"
					pass
				finally:
					readline = f.readline()

dsTraining = parseDataset(".\\data\\TrainingRidge.csv")
dsValidation = parseDataset(".\\data\\ValidationRidge-Y.csv")

# H9.2.A
X = dsTraining.seqVecX_Tp
Y = []
maxSeqY, minSeqY = max(dsTraining.seqY), min(dsTraining.seqY)
for y in dsTraining.seqY:
	nmdY = (y - minSeqY)/(maxSeqY - minSeqY)
	Y.append(nmdY)

clf = NuSVR(C=1.0, nu=0.1)
clf.fit(dsTraining.seqVecX, dsTraining.seqY)  


XValid = dsValidation.seqVecX_Tp
seqYEstm = []
for recX in dsValidation.seqVecX:
	yEstm = clf.predict([recX])
	seqYEstm.append(yEstm)

maxSeqY, minSeqY = max(dsValidation.seqY), min(dsValidation.seqY)
Y = []
for y in dsValidation.seqY:
	nmdY = (y - minSeqY)/(maxSeqY - minSeqY)
	Y.append(nmdY)

# Plot results
plt.clf()
plt.scatter(XValid[0], XValid[1], c=seqYEstm, cmap=plt.get_cmap('coolwarm'))
plt.show()

plt.clf()
plt.scatter(XValid[0], XValid[1], c=Y, cmap=plt.get_cmap('coolwarm'))
plt.show()

# H9.2.B
def funcMSE(seqEstm, seqTrue):
	sumErrSqr = 0
	for rec in zip(seqEstm, seqTrue):
		sumErrSqr += 0.5*(rec[0] - rec[1])**2
	meanErrSqr = sumErrSqr/len(seqEstm)
	return meanErrSqr

C_range = np.logspace(-2, 12, num="15", base=2.0)
gamma_range = np.logspace(-12, 0, num="13", base=2.0)
# C_range = np.logspace(-2, 8, num="6", base=2.0)
# gamma_range = np.logspace(-12, 0, num="7", base=2.0)

seqCG = []
scores = []
scoresLin = []
for C in C_range:
	elemScores = []
	for gamma in gamma_range:
		print(str(C), str(gamma))
		clf = NuSVR(C=C, gamma=gamma, kernel='rbf', nu=0.5)
		clf.fit(dsTraining.seqVecX, dsTraining.seqY)
		seqEstmY = []
		for vecX in dsTraining.seqVecX:
			seqEstmY.append(clf.predict([vecX]))
		meanErrSqr = funcMSE(seqEstmY, dsTraining.seqY)
		elemScores.append(meanErrSqr)
		print(meanErrSqr)
		seqCG.append([C,gamma])
		scoresLin.append(meanErrSqr)
	scores.append(meanErrSqr)

seqCG_Tp = makeTranspose(seqCG)
plt.clf()
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.xlabel('C')
plt.ylabel('gamma')

plt.scatter(seqCG_Tp[0], seqCG_Tp[1], norm=matplotlib.colors.LogNorm(), c=scoresLin, cmap=plt.get_cmap('coolwarm'))
plt.show()

print(min(scoresLin))
print(scoresLin.index(min(scoresLin)))
print(seqCG[scoresLin.index(min(scoresLin))])

# H9.3.C
X = dsTraining.seqVecX_Tp
Y = []
maxSeqY, minSeqY = max(dsTraining.seqY), min(dsTraining.seqY)
for y in dsTraining.seqY:
	nmdY = (y - minSeqY)/(maxSeqY - minSeqY)
	Y.append(nmdY)

clf = NuSVR(C=64.0, nu=0.25)
clf.fit(dsTraining.seqVecX, dsTraining.seqY)  

XValid = dsValidation.seqVecX_Tp
seqYEstm = []
for recX in dsValidation.seqVecX:
	yEstm = clf.predict([recX])
	seqYEstm.append(yEstm)

maxSeqY, minSeqY = max(dsValidation.seqY), min(dsValidation.seqY)
Y = []
for y in dsValidation.seqY:
	nmdY = (y - minSeqY)/(maxSeqY - minSeqY)
	Y.append(nmdY)

# Plot results
plt.clf()
plt.scatter(XValid[0], XValid[1], c=seqYEstm, cmap=plt.get_cmap('coolwarm'))
plt.show()

plt.clf()
plt.scatter(XValid[0], XValid[1], c=Y, cmap=plt.get_cmap('coolwarm'))
plt.show()