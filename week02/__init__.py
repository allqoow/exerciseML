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
#matplotlib.pyplot.scatter(x1List,x2List, c=yList)
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
	matplotlib.pyplot.scatter(x1List,x2List, c=yList)
	x1Vec = [wVec[0]*(-2),0,wVec[0]*2]
	x2Vec = [wVec[1]*(-2),0,wVec[1]*2]
	matplotlib.pyplot.plot(x1Vec, x2Vec)
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