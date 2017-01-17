#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161121(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)

# H5.1.A
import matplotlib.pyplot
import math
import numpy.random
import numpy.linalg


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

def doCentering(data, flag):
	seqX1 = []
	seqX2 = []
	seqObs = []
	# with open("TrainingRidge.csv","r") as f:
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

	# Transpose
	seqXX = makeTranspose(zip(seqX1, seqX2))

	# Sample mean of seqX1
	smX1 = reduce(lambda tmp, x1: tmp+x1, seqX1)/len(seqX1)
	smX2 = reduce(lambda tmp, x2: tmp+x2, seqX2)/len(seqX2)
	seqX1Centered = list(map(lambda x1: round(x1-smX1,2), seqX1))
	seqX2Centered = list(map(lambda x2: round(x2-smX2,1), seqX2))

	seqXXCentered = makeTranspose(zip(seqX1Centered,seqX2Centered))
	return seqXXCentered

def doSphering(data, flag):
	seqX1 = []
	seqX2 = []
	seqObs = []
	# with open("TrainingRidge.csv","r") as f:
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

	# Transpose
	seqXX = makeTranspose(zip(seqX1, seqX2))

	# Sample mean of seqX1
	smX1 = reduce(lambda tmp, x1: tmp+x1, seqX1)/len(seqX1)
	smX2 = reduce(lambda tmp, x2: tmp+x2, seqX2)/len(seqX2)
	seqX1Centered = list(map(lambda x1: round(x1-smX1,2), seqX1))
	seqX2Centered = list(map(lambda x2: round(x2-smX2,1), seqX2))

	seqXXCentered = makeTranspose(zip(seqX1Centered,seqX2Centered))

	cVec=(0,0,1,1) # blue
	# matplotlib.pyplot.scatter(seqXX[0], seqXX[1], c=cVec)
	# matplotlib.pyplot.show()
	# matplotlib.pyplot.clf()

	# matplotlib.pyplot.scatter(seqXXCentered[0], seqXXCentered[1], c=cVec)
	# matplotlib.pyplot.show()
	# matplotlib.pyplot.clf()

	matC = doMatrixMultiplication(seqXXCentered, makeTranspose(seqXXCentered))
	matC = doScalarMultiplication(matC, 0.005)
	print matC

	eigvals = numpy.linalg.eigvals(matC)

	print numpy.linalg.eig(matC)
	matV = numpy.linalg.eig(matC)[1]
	print matV
	matVTranspose = makeTranspose(matV)
	matLambda = [[eigvals[0], 0], [0, eigvals[1]]]
	print matLambda

	print "\n"
	print matC
	result = doMatrixMultiplication(matV,matLambda)
	result = doMatrixMultiplication(result,matVTranspose)
	print result


	# eigVec1 = matVTranspose[0]
	# eigVec2 = matVTranspose[1]
	eigVec1 = matV[0]
	eigVec2 = matV[1]
	tmp1 = []
	tmp2 = []
	for x in makeTranspose(seqXXCentered):
		tmp1.append(makeInnerProduct(eigVec1, x))
		tmp2.append(makeInnerProduct(eigVec2, x))
	seqXXDecorr = [tmp1, tmp2]
	print getNumberRows(seqXXDecorr)
	print getNumberCols(seqXXDecorr)

	# matplotlib.pyplot.scatter(seqXXDecorr[0], seqXXDecorr[1], c=(1,0,0,1))
	# matplotlib.pyplot.show()
	# matplotlib.pyplot.clf()

	tmp1 = []
	tmp2 = []
	someVec1 = [matLambda[0][0]**(-0.5), 0]
	someVec2 = [0, matLambda[1][1]**(-0.5)]
	for x in makeTranspose(seqXXDecorr):
		tmp1.append(makeInnerProduct(someVec1, x))
		tmp2.append(makeInnerProduct(someVec2, x))
	seqXXWhiten = [tmp1, tmp2]
	
	matLambda05 = [someVec1, someVec2]
	matTransform = doMatrixMultiplication(matLambda05, matVTranspose)
	return matTransform
	# matplotlib.pyplot.scatter(seqXXWhiten[0], seqXXWhiten[1], c=cVec)
	# matplotlib.pyplot.show()
	# matplotlib.pyplot.clf()


# Start of Validation
matTransform = doSphering("TrainingRidge.csv","r")
print matTransform
seqXXCenteredValid = doCentering("ValidationRidge-Y.csv","r")


seqXXWhitenValid = doMatrixMultiplication(matTransform, seqXXCenteredValid)
print seqXXWhitenValid

cVec=(0,0,1,1) # blue
# H5.2.B.

# matplotlib.pyplot.scatter(seqXXWhitenValid[0], seqXXWhitenValid[1], c=cVec)
# matplotlib.pyplot.show()
# matplotlib.pyplot.clf()

vv = readCSV("ValidationRidge-Y.csv","r")

for k in range(4):
	for l in range(k+1):
		alpha = l
		beta = k - l
		seqMonom =[]
		seqEstm = []
		for i in range(getNumberCols(seqXXWhitenValid)):
			monom = (seqXXWhitenValid[0][i]**alpha)*(seqXXWhitenValid[1][i]**beta)
			estm = vv[i][2]
			seqMonom.append(monom)
			seqEstm.append(estm)
		print str(alpha) + " " + str(beta)
		matplotlib.pyplot.scatter(seqMonom, seqEstm, c=cVec)
		matplotlib.pyplot.show()
		matplotlib.pyplot.clf()



# def getEigenvalues(mat):
# 	dim = getNumberRows(mat)
# 	mat1 = []
# 	for i in range(dim):
# 		tmpRow = []
# 		for j in range(dim):
# 			if i==j:
# 				elem = mat[i][j] - eigVal
# 			else:
# 				elem = mat[i][j]
# 			tmpRow.append(elem)
# 		mat1.append(mat1)

# def getDeterminant22(mat):
# 	pass
