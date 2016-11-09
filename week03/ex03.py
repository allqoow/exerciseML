#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161107(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)	

# H3.2.A
print "H3.2.A"
import matplotlib.pyplot
import math
import numpy.random

with open("RegressionData.txt","r") as f:
	seqX = []
	seqT = []
	readline = f.readline()
	while readline:
		x = float(readline.strip().split(" ")[0])
		t = float(readline.strip().split(" ")[1])
		seqX.append(x)
		seqT.append(t)
		readline = f.readline()
#print seqX
#print seqT
dataSet = zip(seqX, seqT)

class VecParamSet():
	pass

vecWeight10_i0 = numpy.random.uniform(-0.5, 0.5, 3)
vecWeight10_i1 = numpy.random.uniform(-0.5, 0.5, 3)
vecWeight21_1i = numpy.random.uniform(-0.5, 0.5, 3)

# dict zip?
seqVecParamFunc1 = zip(vecWeight10_i1, vecWeight10_i0)
seqVecParamFunc2 = zip(vecWeight21_1i)
seqSetVecParam = zip(seqVecParamFunc1, seqVecParamFunc2)
print seqSetVecParam
print "\n"

# transfer function at layer 1
def func1(vecX, vecParam):
	weight10_i0 = vecParam[0]
	weight10_i1 = vecParam[1]
	
	x = vecX
	ret = math.tanh(weight10_i1*x-weight10_i0)
	#return round(ret, 4)
	return round(ret, 10)
	#return ret

# transfer function at layer 2
def func2(vecX, vecParam):
	weight21_1i = vecParam[0]

	x = vecX
	ret = weight21_1i*x
	#return round(ret, 4)
	return round(ret, 10)
	#return ret

def funcCost(estm, true):
	ret = 0.5*((estm - true)**2)
	return ret

def pd_funcCost_w10_i0(err, x, setVecParam):
	vecParamFunc1 = setVecParam[0]
	vecParamFunc2 = setVecParam[1]
	weight10_i0 = vecParamFunc1[0]
	weight10_i1 = vecParamFunc1[1]
	weight21_1i = vecParamFunc2[0]
	ret = err*weight21_1i*(math.tanh(weight10_i1*x-weight10_i0)**2-1)
	return ret

def pd_funcCost_w10_i1(err, x, setVecParam):
	vecParamFunc1 = setVecParam[0]
	vecParamFunc2 = setVecParam[1]
	weight10_i0 = vecParamFunc1[0]
	weight10_i1 = vecParamFunc1[1]
	weight21_1i = vecParamFunc2[0]
	ret = err*weight21_1i*(1-math.tanh(weight10_i1*x-weight10_i0))*x
	return ret

def pd_funcCost_w21_1i(err, x, setVecParam):
	vecParamFunc1 = setVecParam[0]
	vecParamFunc2 = setVecParam[1]
	weight10_i0 = vecParamFunc1[0]
	weight10_i1 = vecParamFunc1[1]
	weight21_1i = vecParamFunc2[0]
	ret = err*weight21_1i*math.tanh(weight10_i1*x-weight10_i0)
	return ret

for tryCount in range(3000):
	costErr = 0
	seqEstm = []
	for rec in dataSet:
		x = rec[0]
		t = rec[1]
		
		estm = 0
		for setVecParam in seqSetVecParam:
			vecParamFunc1 = setVecParam[0]
			vecParamFunc2 = setVecParam[1]

			retFunc1 = func1(x, vecParamFunc1)
			retFunc2 = func2(retFunc1, vecParamFunc2)
			estm += retFunc2	
		seqEstm.append(estm)

		# errCost calculated by the quadratic function
		costErr += funcCost(estm, t)

	#print seqEstm
	print "Try " + str(tryCount) + " : "
	print costErr

	# Backward propagation. i.e. updating weights
	dataSet = zip(seqX, seqT, seqEstm)

	vecWeight10_i0Updated = []
	vecWeight10_i1Updated = []
	vecWeight21_1iUpdated = []

	for i in range(len(seqSetVecParam)):
		setVecParam = seqSetVecParam[i]
		n = len(dataSet)

		delta_w10_i0 = 0
		for rec in dataSet:
			err = rec[2]-rec[1]
			x = rec[0]
			delta_w10_i0 += pd_funcCost_w10_i0(err, x, setVecParam)
		delta_w10_i0 = -(delta_w10_i0/n)
		#print seqVecParamFunc2[i][0]
		weight10_i0Updated = seqVecParamFunc1[i][0] + 0.5*delta_w10_i0
		vecWeight10_i0Updated.append(weight10_i0Updated)

		delta_w10_i1 = 0
		for rec in dataSet:
			err = rec[2]-rec[1]
			x = rec[0]
			delta_w10_i1 += pd_funcCost_w10_i1(err, x, setVecParam)
		delta_w10_i1 = -(delta_w10_i1/n)
		#print seqVecParamFunc2[i][0]
		weight10_i1Updated = seqVecParamFunc1[i][1] + 0.5*delta_w10_i1
		vecWeight10_i1Updated.append(weight10_i1Updated)

		delta_w21_1i = 0
		for rec in dataSet:
			err = rec[2]-rec[1]
			x = rec[0]
			delta_w21_1i += pd_funcCost_w21_1i(err, x, setVecParam)
		delta_w21_1i = -(delta_w21_1i/n)
		#print seqVecParamFunc2[i][0]
		weight21_1iUpdated = seqVecParamFunc2[i][0] + 0.5*delta_w21_1i
		vecWeight21_1iUpdated.append(weight21_1iUpdated)

	seqVecParamFunc1 = zip(vecWeight10_i1Updated, vecWeight10_i0Updated)
	seqVecParamFunc2 = zip(vecWeight21_1iUpdated)
	seqSetVecParam = zip(seqVecParamFunc1, seqVecParamFunc2)
	#print seqSetVecParam
	#print "\n"

#pp = matplotlib.pyplot.plot(alphaList, thetaList, c=(1,0,0,1)) # red
#print pp
# 	y1EstList = []
# 	y2EstList = []
# 	for x in xList:
# 		y1Est = 0
# 		y2Est = 0
# 		for vecParam in vecParamList:
# 			wi = VecParam[0]
# 			a1i = VecParam[1]
# 			a2i = VecParam[2]
# 			bi = VecParam[3]
# 			y1Est += wi*math.tanh(a1i*(x-bi))
# 			y2Est += wi*math.tanh(a2i*(x-bi))
# 		y1EstList.append(y1Est)
# 		y2EstList.append(y2Est)

# 	# plotting the estimated functions
# 	matplotlib.pyplot.plot(xList, y1EstList, c=(1,0,0,1)) # red
# 	matplotlib.pyplot.plot(xList, y2EstList, c=(0,0,1,1)) # blue

# 	# assembling generated numbers before clearing
# 	y1EstListList.append(y1EstList)
# 	y2EstListList.append(y2EstList)
# 	VecParamListList.append(VecParamList)

# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()