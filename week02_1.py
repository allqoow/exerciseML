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
colourList = []
with open("applesOranges.csv","r") as openedF:
	rawContent = openedF.read()
	rawContentByLine = rawContent.split("\n")

	for obs in rawContentByLine[1:]: 
		# For stability reason. It operates only if the input data are good.
		if len(obs.split(",")) == 3: 
			splitRec = obs.split(",")
			x1List.append(float(splitRec[0]))
			x2List.append(float(splitRec[1]))
			yList.append(int(splitRec[2]))
			if int(splitRec[2]) == 0:
				colourList.append((1,0,0,1)) # red for apples
			elif int(splitRec[2]) == 1:
				colourList.append((0,0,1,1)) # orange for oranges

print len(colourList)
matplotlib.pyplot.scatter(x1List,x2List, c=colourList)
matplotlib.pyplot.show()

# H2.1.B
print "\nH2.1.B"
pi = math.pi
wVecList = []
for alpha in range(20):
	wVecList.append([math.sin(pi*alpha/20), math.cos(pi*alpha/20)])
	
#print wVecList
obsVecList = zip(x1List, x2List, yList)
sampleSize = float(len(obsVecList))

bestPerformance = 0
for wVec in wVecList:
	countCorrect = 0
	for obsVec in obsVecList:
		if wVec[0]*obsVec[0] + wVec[1]*obsVec[1] > 0:
			est = 1
		elif wVec[0]*obsVec[0] + wVec[1]*obsVec[1] < 0:
			est = 0
		if est == int(obsVec[2]):
			countCorrect += 1

	# evaluation of performance
	performance = countCorrect/sampleSize
	print str(wVec) + " => " + str(performance)
	if bestPerformance < performance:
		bestWVec = wVec
		bestPerformance = performance

	# plotting
	matplotlib.pyplot.scatter(x1List,x2List, c=colourList)
	x2Vec = [-wVec[0]*(-2),0,-wVec[0]*2]
	x1Vec = [wVec[1]*(-2),0,wVec[1]*2]

	matplotlib.pyplot.plot(x1Vec, x2Vec)
	#matplotlib.pyplot.show()
	
# H2.1.C
print "\nH2.1.C"
print str(bestWVec) + " => " + str(performance)
thetaList = [-3 + (x/10.0) for x in range(61)]

bestPerformance = 0
for theta in thetaList:
	countCorrect = 0
	inputText = ""
	for obsVec in obsVecList:
		if bestWVec[0]*obsVec[0] + bestWVec[1]*obsVec[1] + theta > 0:
			est = 1
		elif bestWVec[0]*obsVec[0] + bestWVec[1]*obsVec[1] + theta < 0:
			est = 0
		if est == int(obsVec[2]):
			countCorrect += 1
		#print str(obsVec[0]) +","+str(obsVec[1])+","+str(est)+","+str(obsVec[2])
		inputText += str(obsVec[0]) +","+str(obsVec[1])+","+str(est)+"\n"
		#print inputText
	performance = countCorrect/sampleSize
	print str(theta) + " => " + str(performance)
	if bestPerformance < performance:
		bestTheta = theta
		bestPerformance = performance
		bestInputText = inputText

print bestWVec
print bestTheta

# H2.1.D
with open("applesOrangesEst.txt","w") as res:
	alphaList =range(20)
	thetaList = [-3 + (x/10.0) for x in range(61)]
	writeStr = ""
	for obsVec in obsVecList:
		if bestWVec[0]*obsVec[0] + bestWVec[1]*obsVec[1] + bestTheta> 0:
			est = 1
		elif bestWVec[0]*obsVec[0] + bestWVec[1]*obsVec[1] + bestTheta < 0:
			est = 0
		if est == int(obsVec[2]):
			countCorrect += 1
		writeStr += str(obsVec[0]) +","+ str(obsVec[1])+","+str(est)+"\n"
	res.write(writeStr)

with open("applesOrangesEst.txt","r") as openedF:
	x1List2 = []
	x2List2 = []
	yList2 = []
	colourList2 = []
	
	rawContent = openedF.read()
	rawContentByLine = rawContent.split("\n")
	for obs in rawContentByLine: 
		# For stability reason. It operates only if the input data are good.
		if len(obs.split(",")) == 3: 
			splitRec = obs.strip().split(",")
			x1List2.append(float(splitRec[0]))
			x2List2.append(float(splitRec[1]))
			#yList2.append(int(splitRec[2]))
			if int(splitRec[2]) == 0:
				colourList2.append((1,0,0,1)) # red for apples
			elif int(splitRec[2]) == 1:
				colourList2.append((1,0.5,0,1)) # orange for oranges
	
	bestx2Vec = [(-bestWVec[0])*(-2)-bestTheta,0-bestTheta,(-bestWVec[0])*2-bestTheta]
	bestx1Vec = [bestWVec[1]*(-2),0,bestWVec[1]*2]

	matplotlib.pyplot.clf()
	matplotlib.pyplot.scatter(x1List2,x2List2, c=colourList2)
	matplotlib.pyplot.plot(bestx1Vec, bestx2Vec)
	matplotlib.pyplot.show()

# H2.1.E
with open("results.txt","w") as res:
	alphaList =range(360)
	thetaList = [-3 + (x/40.0) for x in range(241)]
	performanceList = []
	for alpha in alphaList:
		wVec = [math.sin(pi*alpha/80), math.cos(pi*alpha/80)]
		for theta in thetaList:
			countCorrect = 0
			for obsVec in obsVecList:
				if wVec[0]*obsVec[0] + wVec[1]*obsVec[1] + theta> 0:
					est = 0
				elif wVec[0]*obsVec[0] + wVec[1]*obsVec[1] + theta < 0:
					est = 1
				if est == int(obsVec[2]):
					countCorrect += 1

			# evaluation of performance
			performance = countCorrect/sampleSize
			#print "(alpha=" + str(alpha) + ", theta=" + str(theta) + ") => " + str(performance)
			performanceList.append(performance)
			writeStr = str(alpha) +","+ str(theta)+","+str(performance)+"\n"
			res.write(writeStr)

data = numpy.genfromtxt('results.txt',delimiter=',')
alphas=numpy.unique(data[:,0])
thetas=numpy.unique(data[:,1])
Alphas,Thetas = numpy.meshgrid(alphas,thetas)

Performances=data[:,2].reshape(len(thetas),len(alphas))
matplotlib.pyplot.pcolormesh(Alphas,Thetas,Performances)
matplotlib.pyplot.show()

# H2.1.F
# No.
# What if there is a non-linear border (or a borderlike something) between classes?
# We cannot distinguish those (two) classes with a line or hyperplane.

# clearing existing data
matplotlib.pyplot.clf()