#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161129(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)

import math
import random
import sys
# sys.path.insert(0, '../packageML/')

import matplotlib.pyplot
import numpy.random
import numpy.linalg
from sklearn.cluster import KMeans

from moduleML import *

# H7.2
def funcCost(estm, true):
	ret = 0.5*((estm - true)**2)
	return ret

def chooseColour(value):
	if value == 1:
		ret = (1,0,0,1)
	elif value == -1:
		ret = (0,0,1,1)
	return ret

def pd_funcCost(estm, weight):
	ret = estm*weight
	return ret

def pd_funcCost_w2(estm, vecWeight):
	ret = estm*vecWeight[1]
	return ret

def pd_funcCost_b(estm, b):
	ret = estm*b
	return ret

# H7.2
# Generating population
N = 1000
N2 = N/2
seqVecX1C1_Tp_1xN2 = [list(numpy.random.normal(0, 2**0.5, N2))]
seqVecX2C1_Tp_1xN2 = [list(numpy.random.normal(1, 2**0.5, N2))]
seqScalarYC1_Tp_1xN2 = [[1]*N2]
seqColourC1_Tp_1xN2 = [[(1,0,0,1)]*N2]
seqC1_Tp_4xN2 = seqVecX1C1_Tp_1xN2 + seqVecX2C1_Tp_1xN2 + seqScalarYC1_Tp_1xN2 + seqColourC1_Tp_1xN2
seqC1_N2x4 = makeTranspose(seqC1_Tp_4xN2)

seqVecX1C2_Tp_1xN2 = [list(numpy.random.normal(1, 2**0.5, N2))]
seqVecX2C2_Tp_1xN2 = [list(numpy.random.normal(0, 2**0.5, N2))]
seqScalarYC2_Tp_1xN2 = [[1]*N2]
seqColourC2_Tp_1xN2 = [[(0,1,0,1)]*N2]
seqC2_Tp_4xN2 = seqVecX1C2_Tp_1xN2 + seqVecX2C2_Tp_1xN2 + seqScalarYC2_Tp_1xN2 + seqColourC2_Tp_1xN2
seqC2_N2x4 = makeTranspose(seqC2_Tp_4xN2)

seqData_Nx4 = seqC1_N2x4 + seqC2_N2x4
seqData_Tp_4xN = makeTranspose(seqData_Nx4)

# matplotlib.pyplot.scatter(seqData_Tp_4xN[0], seqData_Tp_4xN[1], c=seqData_Tp_4xN[3], lw=1)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

seqSampleSize = [2,4,6,8,10,20,40,100]
# seqSampleSize = range(2,101)
seqMeanRTrain = []
seqStdRTrain = []
seqMeanRTest = []
seqStdRTest = []
seqMeanVecWeight0 = []
seqMeanVecWeight1 = []
seqMeanVecWeight2 = []
seqStdVecWeight0 = []
seqStdVecWeight1 = []
seqStdVecWeight2 = []
for sampleSize in seqSampleSize:

	vecWeight = [random.random(), random.random(), random.random()]

	eta = 1
	sumCostPrev = 10000	
	sumCost = 1
	seqRTrain = []
	seqRTest = []
	seqVecWeight = []
	for i in range(50):
		seqData_NSampx4 = makeSample(seqData_Nx4, sampleSize)
		
		seqHatY_1xNSamp = []
		sumCost = 0
		for rec in seqData_NSampx4:
			vecX = [1] + list(rec[0:2])
			scalarY = rec[2]
			estm = makeInnerProduct(vecWeight, vecX)
			err = funcCost(estm, scalarY)
			sumCost += err
			if estm > 0:
				seqHatY_1xNSamp.append(1)
			elif estm < 0:
				seqHatY_1xNSamp.append(-1)

		delta_w0 = err*(-eta*pd_funcCost(estm, vecWeight[0]))/len(seqData_NSampx4)
		delta_w1 = err*(-eta*pd_funcCost(estm, vecWeight[1]))/len(seqData_NSampx4)
		delta_w2 = err*(-eta*pd_funcCost(estm, vecWeight[2]))/len(seqData_NSampx4)

		if sumCostPrev > sumCost:
			eta = 1.2*eta

			vecWeight[0] += delta_w0
			vecWeight[1] += delta_w1
			vecWeight[2] += delta_w2
			vecWeightPrev = vecWeight
		elif sumCostPrev < sumCost:
			eta = 0.8*eta

		seqData_4xNSamp = makeTranspose(seqData_NSampx4)
		rTrain = (makeInnerProduct(seqData_4xNSamp[2],seqHatY_1xNSamp) + sampleSize)/float(sampleSize)
		seqRTrain.append(rTrain)

		seqHatY_1xN = []
		for rec in seqData_Nx4:
			vecX = [1] + list(rec[0:2])
			scalarY = rec[2]
			estm = makeInnerProduct(vecWeight, vecX)
			err = funcCost(estm, scalarY)
			sumCost += err
			if estm > 0:
				seqHatY_1xN.append(1)
			elif estm < 0:
				seqHatY_1xN.append(-1)

		seqData_4xN = makeTranspose(seqData_Nx4)
		rTest = (makeInnerProduct(seqData_4xN[2],seqHatY_1xN) + 1000)/float(1000)
		seqRTest.append(rTest)

		sumCostPrev = float(sumCost)
		seqVecWeight.append(vecWeight)

	seqMeanRTrain.append(numpy.mean(seqRTrain))
	seqStdRTrain.append(numpy.std(seqRTrain))
	seqMeanRTest.append(numpy.mean(seqRTest))
	seqStdRTest.append(numpy.std(seqRTest))

	seqVecWeight_Tp = makeTranspose(seqVecWeight)
	
	seqMeanVecWeight0.append(numpy.mean(seqVecWeight_Tp[0]))
	seqMeanVecWeight1.append(numpy.mean(seqVecWeight_Tp[1]))
	seqMeanVecWeight2.append(numpy.mean(seqVecWeight_Tp[2]))
	seqStdVecWeight0.append(numpy.std(seqVecWeight_Tp[0]))
	seqStdVecWeight1.append(numpy.std(seqVecWeight_Tp[1]))
	seqStdVecWeight2.append(numpy.std(seqVecWeight_Tp[2]))

	# bestx2Vec = [vecWeight[1]*(-1000)-vecWeight[0],vecWeight[1]*1000-vecWeight[0]]
	# bestx1Vec = [vecWeight[2]*(-1000),vecWeight[2]*1000]

	# matplotlib.pyplot.axis([-6, 6, -6, 6])
	# matplotlib.pyplot.scatter(seqData_Tp_4xN[0], seqData_Tp_4xN[1], c=seqData_Tp_4xN[3], lw=1)
	# matplotlib.pyplot.plot(bestx1Vec, bestx2Vec)
	# matplotlib.pyplot.show()
	# matplotlib.pyplot.clf()

print seqMeanVecWeight0

# H7.2.a
# matplotlib.pyplot.axis([0, 100, -5, 5])
matplotlib.pyplot.scatter(seqSampleSize, seqMeanRTrain, color=(1,0,0,1))
matplotlib.pyplot.scatter(seqSampleSize, seqMeanRTest, color=(0,0,1,1))
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# matplotlib.pyplot.axis([0, 100, -5, 5])
matplotlib.pyplot.bar(seqSampleSize, seqStdRTrain, color=(1,0,0,1))
matplotlib.pyplot.bar(seqSampleSize, seqStdRTest, color=(0,0,1,1))
matplotlib.pyplot.show()
matplotlib.pyplot.clf()

# H7.2.b
# matplotlib.pyplot.axis([0, 100, -5, 5])
# matplotlib.pyplot.bar(seqSampleSize, seqMeanVecWeight1)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# matplotlib.pyplot.axis([0, 100, -5, 5])
# matplotlib.pyplot.bar(seqSampleSize, seqMeanVecWeight2)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# matplotlib.pyplot.axis([0, 100, -5, 5])
# matplotlib.pyplot.bar(seqSampleSize, seqMeanVecWeight0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# matplotlib.pyplot.bar(seqSampleSize, seqStdVecWeight1)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# matplotlib.pyplot.bar(seqSampleSize, seqStdVecWeight2)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# matplotlib.pyplot.bar(seqSampleSize, seqStdVecWeight0)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# H7.2.c
# No apparent trend across N for mean,
# decresing trend acroos N for std,
# which implies stable convergence to the best weight when N is large
# some outlying result around 20

# H7.3
n = 100
prob = 0.4
lambd = n*prob

seqK = []
seqBinomK = []
seqNormalK = []
seqPoissonK = []
for k in range(n+1):
	seqK.append(k)
	seqBinomK.append(funcBinom(k,n,prob))
	seqNormalK.append(funcGaussian(k,prob*n,prob*(1-prob)*n))
	seqPoissonK.append(funcPoisson(k,lambd))

# matplotlib.pyplot.plot(seqK, seqBinomK, c=(1,0,0,1), lw=1)
# matplotlib.pyplot.plot(seqK, seqNormalK, c=(0,0,1,1), lw=1)
# matplotlib.pyplot.plot(seqK, seqPoissonK, c=(0,1,0,1), lw=1)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

# H7.3.a
# How many times you might get checked in the S- or U-Bahn given you ride N times?
# Target value is governed by probability and discrete manner.
# When something is determined in continuous manner.
# e.g. Price for single journey ticket across difference cities in the world

# H7.3.b
# When n is large, prob is close to 0.5
# Because when n is large, skewness is not that large... CLT?
# n = 100, prob = 0.4
# No.

# H7.3.c
# When n is small, prob is not close to 0.5
# n = 10, prob = 0.05
# Yes.

