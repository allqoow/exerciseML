#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161020(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)	

from PIL import Image

sampleDir = ".\\samples\\"
print dir(Image)
openedImg = Image.open(sampleDir + "sample0.jpg", "r")
loadedImg = openedImg.load()
pixels = list(openedImg.getdata())
print openedImg.size
print loadedImg[2000-1,1125-1]

filterVec = [
	-1,-1,-1,
	-1, 8,-1,
	-1,-1,-1
]

filterWidth = 100
scale = 33
anchors = [
	(anchorX*filterWidth,anchorY*filterWidth) 
	for anchorX in range(2000/filterWidth)
	for anchorY in range(1125/filterWidth)
]
print anchors


for anchorVec in anchors:
	coors = [
		(xcoor,ycoor)
		for xcoor in range(anchorVec[0],anchorVec[0]+(filterWidth/scale)*scale,scale)
		for ycoor in range(anchorVec[1],anchorVec[1]+(filterWidth/scale)*scale,scale)
	]
	#print coors
	obsVec = []
	for v in coors:
		sumR = 0
		for i in range(scale):
			sumR += loadedImg[v[0]+i,v[1]+i][0]
		obsVec.append(sumR)

	#print filterVec
	#print obsVec
	indicator = 0
	for v in zip(filterVec, obsVec):
		#print v[0], "  ", v[1]
		indicator += v[0]*v[1]
	print str(indicator) + "  " + str(anchorVec)

def applyFilter(filterName, scale):
	pass
	#for xcor in range(2000/scale):	
	#filterName