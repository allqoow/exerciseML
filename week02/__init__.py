#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161029(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)	

# H2.1.A
print "H2.1.A"
import matplotlib.pyplot
import math
import numpy.random

x1List = []
x2List = []
yList = []
with open("applesOranges.csv","r") as openedF:
	rawContent = openedF.read()
	rawContentByLine = rawContent.split("\n")

	for obs in rawContentByLine[1:]:
		if len(obs.split(",")) == 3:
			splitRec = obs.split(",")
			x1List.append(float(splitRec[0]))
			x2List.append(float(splitRec[1]))
			yList.append(int(splitRec[2]))

# blue for y=0
matplotlib.pyplot.scatter(x1List,x2List, c=yList)
#matplotlib.pyplot.show()

# H2.1.B
print "\nH2.1.B"
pi = math.pi
wVecList = []
for x in range(20):
	wVecList.append([math.sin(pi*x/20), math.cos(pi*x/20)])
	
print wVecList
obsVecList = zip(x1List, x2List, yList)
sampleSize = float(len(obsVecList))

bestPerformance = 0
for wVec in wVecList:
	countCorrect = 0
	for obsVec in obsVecList:
		if wVec[0]*obsVec[0] + wVec[1]*obsVec[1] > 0:
			est = 0
		elif wVec[0]*obsVec[0] + wVec[1]*obsVec[1] < 0:
			est = 1
		if est == int(obsVec[2]):
			countCorrect += 1

	# evaluation of performance
	performance = countCorrect/sampleSize
	print str(wVec) + " => " + str(performance)
	if bestPerformance < performance:
		bestWVec = wVec
		bestPerformance = performance

	# plotting
	#matplotlib.pyplot.scatter(x1List,x2List, c=yList)
	x1Vec = [wVec[0]*(-2),0,wVec[0]*2]
	x2Vec = [wVec[1]*(-2),0,wVec[1]*2]
	#matplotlib.pyplot.plot(x1Vec, x2Vec)
	#matplotlib.pyplot.show()
	
# H2.1.C
print "\nH2.1.C"
print str(bestWVec) + " => " + str(performance)
thetaList = []
for x in range(61):
	thetaList.append(-3 + (x/10.0))
print thetaList

bestPerformance = 0
for theta in thetaList:
	countCorrect = 0
	for obsVec in obsVecList:
		if bestWVec[0]*obsVec[0] + bestWVec[1]*obsVec[1] + theta > 0:
			est = 0
		elif bestWVec[0]*obsVec[0] + bestWVec[1]*obsVec[1] + theta < 0:
			est = 1
		if est == int(obsVec[2]):
			countCorrect += 1
	performance = countCorrect/sampleSize
	print str(theta) + " => " + str(performance)
	if bestPerformance < performance:
		bestTheta = theta
		bestPerformance = performance
print bestTheta

# H2.1.F
# No.
# What if there is a non-linear border (or a borderlike something) between classes?
# We cannot distinguish those (two) classes with a line or hyperplane.

# clearing existing data
matplotlib.pyplot.clf()


# H2.2.A
# When data is not linearly seperable, like the case mentioned at H2.1.F.

# H2.2.B
xList = [x*0.1 for x in range(-20,20)]
for i in range(50):
	wList = numpy.random.normal(0, 1, 10)
	aList1 = numpy.random.normal(0, 2, 10)
	bList = numpy.random.uniform(-2, 2, 10)
	paramVecList = zip(wList, aList1, bList)

	y1List = []
	for x in xList:
		y1 = 0
		for paramVec in paramVecList:
			wi = paramVec[0]
			a1i = paramVec[1]
			bi = paramVec[2]
			y1 += wi*math.tanh(a1i*(x-bi))
		y1List.append(y1)
	matplotlib.pyplot.plot(xList, y1List, c=(1,0,0,1))

matplotlib.pyplot.show()
matplotlib.pyplot.clf()

# H2.2.C
xList = [x*0.1 for x in range(-20,20)]
for i in range(50):
	wList = numpy.random.normal(0, 1, 10)
	aList2 = numpy.random.normal(0, 0.5, 10)
	bList = numpy.random.uniform(-2, 2, 10)
	paramVecList = zip(wList, aList2, bList)

	y2List = []
	for x in xList:
		y2 = 0
		for paramVec in paramVecList:
			wi = paramVec[0]
			a2i = paramVec[1]
			bi = paramVec[2]
			y2 += wi*math.tanh(a2i*(x-bi))
		y2List.append(y2)
	matplotlib.pyplot.plot(xList, y2List, c=(0,0,1,1))

matplotlib.pyplot.show()