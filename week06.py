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
from sklearn.cluster import KMeans

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

# H6.1.A
# C1, red
cdfMu1 = genRandomVecs(30, 2**0.5, [0, 1])
cdfMu2 = genRandomVecs(30, 2**0.5, [1, 0])
cdfMu1.append([(1,0,0,1)]*30)
cdfMu2.append([(1,0,0,1)]*30)
cdfMu1 = makeTranspose(cdfMu1)
cdfMu2 = makeTranspose(cdfMu2)
seqXC1 = cdfMu1 + cdfMu2
seqXC1 = makeTranspose(seqXC1)

# C2, blue
cdfMu3 = genRandomVecs(30, 2**0.5, [0, 0])
cdfMu4 = genRandomVecs(30, 2**0.5, [1, 1])
cdfMu3.append([(0,0,1,1)]*30)
cdfMu4.append([(0,0,1,1)]*30)
cdfMu3 = makeTranspose(cdfMu3)
cdfMu4 = makeTranspose(cdfMu4)
seqXC2 = cdfMu3 + cdfMu4
seqXC2 = makeTranspose(seqXC2)

seqX = makeTranspose(makeTranspose(seqXC1)+makeTranspose(seqXC2))
matplotlib.pyplot.scatter(seqX[0], seqX[1], c=seqX[2])
matplotlib.pyplot.show()
matplotlib.pyplot.clf()

# H6.1.B
def makeSeqPixel(resolution):
	pixelSize = 2/float(resolution)
	seqPixel = []
	for x1 in [(-0.5)+x1*pixelSize for x1 in range(resolution)]:
		for x2 in [(-0.5)+x2*pixelSize for x2 in range(resolution)]:
			seqPixel.append([x1,x2])
	return seqPixel

def doKnnClassification(k, resolution):
	seqPixel = makeSeqPixel(resolution)

	seqCateg = []
	for pixel in seqPixel:	
		setKNN = []
		for x in makeTranspose(seqX):	
			if len(setKNN) < k:
				setKNN.append(x)
			else:
				setKNN.append(x)
				setDist = []
				for y in setKNN:
					dist = getEucDist(pixel[:2], y[:2])
					setDist.append(dist)
				i = 0
				for j in range(len(setDist)):
					if setDist[i] < setDist[j]:
						i = j
				del setKNN[i]

		countRed = 0
		countBlue = 0
		for nn in setKNN:
			if nn[2] == (1,0,0,1):
				countRed += 1
			elif nn[2] == (0,0,1,1):
				countBlue += 2
		if countRed > countBlue:
			seqCateg.append((1,0,0,1))
		elif countRed < countBlue:
			seqCateg.append((0,0,1,1))

	seqPixel = makeTranspose(seqPixel)
	seqPixel.append(seqCateg)
	return seqPixel

# seqPixel = doKnnClassification(3, 100)
# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# H6.1.C
def doParzenClassification(resolution, variance):
	seqPixel = makeSeqPixel(resolution)

	seqCateg = []
	for pixel in seqPixel:
		countRed = 0
		countBlue = 0
		for x in makeTranspose(seqX):
			dist = getEucDist(pixel[:2], x[:2])
			weight = math.exp(-(1/(2*variance))*dist)

			if x[2] == (1,0,0,1):
				countRed += weight
			elif x[2] == (0,0,1,1):
				countBlue += weight
		
		if countRed > countBlue:
			seqCateg.append((1,0,0,1))
		elif countRed < countBlue:
			seqCateg.append((0,0,1,1))

	seqPixel = makeTranspose(seqPixel)
	seqPixel.append(seqCateg)
	return seqPixel

# seqPixel = doParzenClassification(100, variance=0.1)
# seqPixel = doParzenClassification(100, variance=0.5)
# seqPixel = doParzenClassification(100, variance=0.01)

# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# C1, green
seqXC3 = genRandomVecs(60, 0.05, [0.5, 0.5])
seqXC3.append([(0,1,0,1)]*60)

seqX = makeTranspose(makeTranspose(seqXC1)+makeTranspose(seqXC2)+makeTranspose(seqXC3))

# matplotlib.pyplot.scatter(seqX[0], seqX[1], c=seqX[2], ls='None')
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

def doKnnClassification3(k, resolution):
	seqPixel = makeSeqPixel(resolution)

	seqCateg = []
	for pixel in seqPixel:	
		setKNN = []
		for x in makeTranspose(seqX):	
			if len(setKNN) < k:
				setKNN.append(x)
			else:
				setKNN.append(x)
				setDist = []
				for y in setKNN:
					dist = getEucDist(pixel[:2], y[:2])
					setDist.append(dist)
				i = 0
				for j in range(len(setDist)):
					if setDist[i] < setDist[j]:
						i = j
				del setKNN[i]

		countRed = 0
		countBlue = 0
		countGreen = 0
		for nn in setKNN:
			if nn[2] == (1,0,0,1):
				countRed += 1
			elif nn[2] == (0,0,1,1):
				countBlue += 2
			elif nn[2] == (0,1,0,1):
				countGreen += 2

		if countRed > countBlue and countRed > countGreen:
			seqCateg.append((1,0,0,1))
		elif countBlue > countRed and countBlue > countGreen:
			seqCateg.append((0,0,1,1))
		elif countGreen > countRed and countGreen > countBlue:
			seqCateg.append((0,1,0,1))

	seqPixel = makeTranspose(seqPixel)
	seqPixel.append(seqCateg)
	return seqPixel

# seqPixel = doKnnClassification3(5, 100)
# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

def doParzenClassification3(k, resolution, variance):
	seqPixel = makeSeqPixel(resolution)

	seqCateg = []
	for pixel in seqPixel:
		countRed = 0
		countBlue = 0
		countGreen = 0
		for x in makeTranspose(seqX):
			dist = getEucDist(pixel[:2], x[:2])
			weight = math.exp(-(1/(2*variance))*dist)

			if x[2] == (1,0,0,1):
				countRed += weight
			elif x[2] == (0,0,1,1):
				countBlue += weight
			elif x[2] == (0,1,0,1):
				countGreen += weight

		if countRed > countBlue and countRed > countGreen:
			seqCateg.append((1,0,0,1))
		elif countBlue > countRed and countBlue > countGreen:
			seqCateg.append((0,0,1,1))
		elif countGreen > countRed and countGreen > countBlue:
			seqCateg.append((0,1,0,1))

	seqPixel = makeTranspose(seqPixel)
	seqPixel.append(seqCateg)
	return seqPixel

# seqPixel = doParzenClassification3(5, 100, variance=0.5)
# # seqPixel = doParzenClassification3(5, 20, variance=0.1)
# # seqPixel = doParzenClassification3(5, 20, variance=0.01)

# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# print makeTranspose(seqX[:2])

# H6.1.D
# Initialisation

def genRandomVec():
	ret = []
	ret.append(-0.5+random.random()*2)
	ret.append(-0.5+random.random()*2)
	return ret

def doRbfClassification(seqPoint, resolution, variance):
	seqPixel = makeSeqPixel(resolution)
	seqCateg = []
	for pixel in seqPixel:
		vecBasis = [1]
		for point in seqPoint:
			dist = getEucDist(pixel[:2], point)
			weight = math.exp(-(1/(2*variance))*dist)
			vecBasis.append(weight)
		funcSgn = makeInnerProduct(vecWeight, vecBasis)
		print funcSgn
		if funcSgn > 0:
			seqCateg.append((1,0,0,1))
		else:
			seqCateg.append((0,0,1,1))

	seqPixel = makeTranspose(seqPixel)
	seqPixel.append(seqCateg)
	return seqPixel

# k = 2
point1 = genRandomVec()
point2 = genRandomVec()

sumDelta = 9999
while sumDelta > 0.01:
	cluster1 = []
	cluster2 = []
	for x in makeTranspose(seqX):
		distTo1 = getEucDist(x[:2], point1)
		distTo2 = getEucDist(x[:2], point2)

		if distTo1 < distTo2:
			cluster1.append(x)
		elif distTo2 < distTo1:
			cluster2.append(x)

	cluster1T = makeTranspose(cluster1)
	cluster2T = makeTranspose(cluster2)

	point1Prev = point1
	point2Prev = point2

	point1 = [sum(cluster1T[0])/len(cluster1T[0]), sum(cluster1T[1])/len(cluster1T[1])]
	point2 = [sum(cluster2T[0])/len(cluster2T[0]), sum(cluster2T[1])/len(cluster2T[1])]

	delta1 = getEucDist(point1Prev,point1)
	delta2 = getEucDist(point2Prev,point2)
	sumDelta = delta1+delta2

seqPoint2 = [point1, point2] 
# matplotlib.pyplot.scatter(cluster1T[0], cluster1T[1], c=(1,0,0,1), lw=0)
# matplotlib.pyplot.scatter(cluster2T[0], cluster2T[1], c=(0,0,1,1), lw=0)
# matplotlib.pyplot.scatter(point1[0], point1[1], c=(1,0,0,1), lw=1)
# matplotlib.pyplot.scatter(point2[0], point2[1], c=(0,0,1,1), lw=1)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

matPhi = []
vecT = []

point1 = seqPoint2[0]
point2 = seqPoint2[1]
for x in makeTranspose(seqX):
	variance =0.5
	distTo1 = getEucDist(x[:2], point1)
	distTo2 = getEucDist(x[:2], point2)
	weight1 = math.exp(-(1/(2*variance))*distTo1)
	weight2 = math.exp(-(1/(2*variance))*distTo2)
	vecPhi = [1, weight1, weight2]
	matPhi.append(vecPhi)

	if x in cluster1:
		vecT.append([1])
	else:
		vecT.append([-1])

pinvPhi = numpy.linalg.pinv(matPhi)
vecWeight = doMatrixMultiplication(pinvPhi, vecT)
vecWeight = makeTranspose(vecWeight)[0]


# seqPixel = doRbfClassification(seqPoint2, 100, 0.5)

# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# seqPixel = doRbfClassification(seqPoint2, 100, 0.1)

# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# k=3
point1 = genRandomVec()
point2 = genRandomVec()
point3 = genRandomVec()

sumDelta = 9999
while sumDelta > 0.01:
	cluster1 = []
	cluster2 = []
	cluster3 = []
	for x in makeTranspose(seqX):
		distTo1 = getEucDist(x[:2], point1)
		distTo2 = getEucDist(x[:2], point2)
		distTo3 = getEucDist(x[:2], point3)

		if distTo1 < distTo2 and distTo1 < distTo3:
			cluster1.append(x)
		elif distTo2 < distTo3 and distTo2 < distTo1:
			cluster2.append(x)
		elif distTo3 < distTo2 and distTo3 < distTo1:
			cluster3.append(x)

	cluster1T = makeTranspose(cluster1)
	cluster2T = makeTranspose(cluster2)
	cluster3T = makeTranspose(cluster3)

	point1Prev = point1
	point2Prev = point2
	point3Prev = point3

	point1 = [sum(cluster1T[0])/len(cluster1T[0]), sum(cluster1T[1])/len(cluster1T[1])]
	point2 = [sum(cluster2T[0])/len(cluster2T[0]), sum(cluster2T[1])/len(cluster2T[1])]
	point3 = [sum(cluster3T[0])/len(cluster3T[0]), sum(cluster3T[1])/len(cluster3T[1])]

	delta1 = getEucDist(point1Prev,point1)
	delta2 = getEucDist(point2Prev,point2)
	delta3 = getEucDist(point3Prev,point3)
	sumDelta = delta1+delta2+delta3

seqPoint3 = [point1, point2, point3] 

matPhi = []
vecT = []

point1 = seqPoint2[0]
point2 = seqPoint2[1]
point3 = seqPoint3[1]
for x in makeTranspose(seqX):
	variance =0.5
	distTo1 = getEucDist(x[:2], point1)
	distTo2 = getEucDist(x[:2], point2)
	distTo3 = getEucDist(x[:2], point3)
	weight1 = math.exp(-(1/(2*variance))*distTo1)
	weight2 = math.exp(-(1/(2*variance))*distTo2)
	weight3 = math.exp(-(1/(2*variance))*distTo3)
	vecPhi = [1, weight1, weight2, weight3]
	matPhi.append(vecPhi)

	if x in cluster1:
		vecT.append([1])
	else:
		vecT.append([-1])

pinvPhi = numpy.linalg.pinv(matPhi)
vecWeight = doMatrixMultiplication(pinvPhi, vecT)
vecWeight = makeTranspose(vecWeight)[0]

# seqPixel = doRbfClassification(seqPoint3, 100, 0.5)

# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# seqPixel = doRbfClassification(seqPoint3, 100, 0.1)

# matplotlib.pyplot.scatter(seqPixel[0], seqPixel[1], c=seqPixel[2], lw=0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()