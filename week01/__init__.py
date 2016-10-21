#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161020(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)	

import glob, os, shutil, time

from PIL import Image

sampleDir = ".\\samples\\"
cddDir = ".\\cdds\\"
img = Image.open(sampleDir + "sample01.jpg", "r")

filterVec = [
	-1,-1,-1,
	-1, 8,-1,
	-1,-1,-1
]

scale = 21

def genAnchors(img, scale, filterVec):
	imgWidth = img.size[0]
	imgHeight= img.size[1]
	anchorIntv = int(scale*len(filterVec)**0.5)
	anchors = [
		(anchorX, anchorY) 
		for anchorX in range(0, imgWidth, anchorIntv)[:-1]
		for anchorY in range(0, imgHeight, anchorIntv)[:-1]
	]
	return anchors

def genScope(anchor, scale, filterVec):
	anchorIntv = int(scale*len(filterVec)**0.5)
	coors = [
		(xcoor,ycoor)
		for xcoor in range(anchor[0], anchor[0]+anchorIntv, scale)
		for ycoor in range(anchor[1], anchor[1]+anchorIntv, scale)
	]
	return coors

def genObsVec(img, scope):
	imgLoaded = img.load()
	obsVec = []
	for v in scope:
		sumRed = 0
		for i in range(scale):
			for j in range(scale):
				sumRed += -(imgLoaded[v[0]+i,v[1]+i][0]/255.0)
		sumRedAdjusted = sumRed/(scale**2)
		obsVec.append(sumRedAdjusted)
	return obsVec

def applyFilter(filterVec, obsVec):
	indicator = 0
	for v in zip(filterVec, obsVec):
		indicator += v[0]*v[1]
	return indicator

def flushCddDir():
	fileList = [ cddDir + f for f in os.listdir(".\\cdds\\") if f.endswith(".jpg") ]
	for f in fileList:
		os.remove(f)

def cropResult(img, indicator):
	anchorIntv = int(scale*len(filterVec)**0.5)
	xcoorUL = anchor[0]
	ycoorUL = anchor[1]
	xcoorLD = anchor[0] + anchorIntv
	ycoorLD = anchor[1] + anchorIntv
	croppingArea = (xcoorUL, ycoorUL, xcoorLD, ycoorLD)
	img.crop(croppingArea).save(cddDir + str(indicator) +".jpg")

# initialising cdds
flushCddDir()

anchors = genAnchors(img, scale, filterVec)
for anchor in anchors:
	scope = genScope(anchor, scale, filterVec)
	obsVec = genObsVec(img, scope)
	indicator = applyFilter(filterVec, obsVec)
	print str(indicator).ljust(20) + str(anchor).rjust(15)

	# saving cuts of interesting trials
	if indicator != 0:
		cropResult(img, indicator)