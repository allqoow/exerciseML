#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161029(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)	

import matplotlib.pyplot
import math
import numpy.random
# H2.2.A
# When data is not linearly seperable. (e.g. the case mentioned at H2.1.F.)

# H2.2.B, H2.2.C
y1EstListList = []
y2EstListList = []
paramVecListList = []
for i in range(50):
	# list of input variables
	xList = [x*0.1 for x in range(-20,20)]

	# lists of parameter variables
	wList = numpy.random.normal(0, 1, 10)
	aList1 = numpy.random.normal(0, 2, 10)
	aList2 = numpy.random.normal(0, 0.5, 10)
	bList = numpy.random.uniform(-2, 2, 10)
	paramVecList = zip(wList, aList1, aList2, bList)
	
	y1EstList = []
	y2EstList = []
	for x in xList:
		y1Est = 0
		y2Est = 0
		for paramVec in paramVecList:
			wi = paramVec[0]
			a1i = paramVec[1]
			a2i = paramVec[2]
			bi = paramVec[3]
			y1Est += wi*math.tanh(a1i*(x-bi))
			y2Est += wi*math.tanh(a2i*(x-bi))
		y1EstList.append(y1Est)
		y2EstList.append(y2Est)

	# plotting the estimated functions
	matplotlib.pyplot.plot(xList, y1EstList, c=(1,0,0,1)) # red
	matplotlib.pyplot.plot(xList, y2EstList, c=(0,0,1,1)) # blue

	# assembling generated numbers before clearing
	y1EstListList.append(y1EstList)
	y2EstListList.append(y2EstList)
	paramVecListList.append(paramVecList)

matplotlib.pyplot.show()
matplotlib.pyplot.clf()

# Functions with ai~N(0,2) are more dispersed.

# H2.4.D
yRealList = [(-1)*x*0.1 for x in range(-20,20)]

# for the former 50 functions
sumSqrErrList = []
for y1EstList in y1EstListList:
	sumSqrErr = 0
	for x in zip(yRealList, y1EstList):
		sumSqrErr += (x[0]-x[1])**2
	sumSqrErrList.append(sumSqrErr)

print min(sumSqrErrList)
indexOpt1 = sumSqrErrList.index(min(sumSqrErrList))
opt1ParamVecList = paramVecListList[indexOpt1]

y1EstList = []
for x in xList:
	y1Est = 0
	y2Est = 0
	for paramVec in opt1ParamVecList:
		wi = paramVec[0]
		a1i = paramVec[1]
		bi = paramVec[3]
		y1Est += wi*math.tanh(a1i*(x-bi))
	y1EstList.append(y1Est)

# plotting the estimated functions
matplotlib.pyplot.plot(xList, y1EstList, c=(1,0,0,1)) # red

# for the latter 50 functions
sumSqrErrList = []
for y2EstList in y2EstListList:
	sumSqrErr = 0
	for x in zip(yRealList, y2EstList):
		sumSqrErr += (x[0]-x[1])**2
	sumSqrErrList.append(sumSqrErr)

print min(sumSqrErrList)
indexOpt2 = sumSqrErrList.index(min(sumSqrErrList))
opt2ParamVecList = paramVecListList[indexOpt2]

y2EstList = []
for x in xList:
	y2Est = 0
	for paramVec in opt2ParamVecList:
		wi = paramVec[0]
		a2i = paramVec[2]
		bi = paramVec[3]
		y2Est += wi*math.tanh(a2i*(x-bi))
	y2EstList.append(y2Est)

# plotting the estimated functions
matplotlib.pyplot.plot(xList, y2EstList, c=(0,0,1,1)) # red
matplotlib.pyplot.show()