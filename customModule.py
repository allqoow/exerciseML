#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161129(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)

import math
import random

import matplotlib.pyplot
import numpy.random
import numpy.linalg

class CustomModule():
	pass

def getNumberRows(mat):
	return len(mat)

def getNumberCols(mat):
	return len(mat[0])

def makeTranspose(mat):
	return zip(*mat)

def makeInnerProduct(vec0, vec1):
	seqTerm = map(lambda x: x[0]*x[1], zip(vec0, vec1))
	return sum(seqTerm)

def doMatrixMultiplication(mat0, mat1):
	ret = None
	if getNumberCols(mat0) == getNumberRows(mat1):
		nRows = getNumberRows(mat0)
		nCols = getNumberCols(mat1)
		mat1Transpose = makeTranspose(mat1)
		ret = []
		for i in range(nRows):
			tmpRow = []
			for j in range(nCols):
				tmpRow.append(makeInnerProduct(mat0[i],mat1Transpose[j]))
			ret.append(tmpRow)
	else:
		pass
	return ret

def doScalarMultiplication(mat, scalar):
	nRows = getNumberRows(mat)
	nCols = getNumberCols(mat)
	ret = []
	for i in range(nRows):
		tmp = []
		for j in range(nCols):
			tmp.append(mat[i][j]*scalar)
		ret.append(tmp)
	return ret

def readCSV(data, flag):
	seqX1 = []
	seqX2 = []
	seqObs = []
	with open(data, flag) as f:	
		readline = f.readline()
		while readline:
			try:
				x1 = float(readline.strip().split(",")[0])
				x2 = float(readline.strip().split(",")[1])
				obs = float(readline.strip().split(",")[2])
				seqX1.append(x1)
				seqX2.append(x2)
				seqObs.append(obs)
				# print readline.strip()
			except ValueError:
				# print "ValueError: This row is the header"
				pass
			finally:
				readline = f.readline()
	return zip(seqX1,seqX2,seqObs)

def genRandomVecs(sampleSize, std, vec):
	ret = []
	for mu in vec:
		ret.append(numpy.random.normal(mu, std, sampleSize))
	return ret

def getEucDist(point0, point1):
	dist = 0
	for xn in zip(point0, point1):
		dist += (xn[0] - xn[1])**2
	dist**0.5
	return dist

def combination(n,k):
	return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

# returns 
def funcBinom(k,n,prob):
	return combination(n,k)*(prob**k)*((1-prob)**(n-k))

def funcGaussian(x, mu, var):
	std = var**0.5
	return 1./(std*(2.*math.pi)**.5)*numpy.exp(-((x - mu)**2)/(2*(std**2)))

def funcPoisson(k,lambd):
	return ((lambd**k)/math.factorial(k))*math.exp(-lambd)

def makeSample(pop, sampleSize):
	ret = []
	seqRandIndex = list(random.sample(range(0, len(pop)), sampleSize))
	for i in seqRandIndex:
		ret.append(pop[i])
	return ret