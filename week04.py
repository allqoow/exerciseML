#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author    : allqoow
# Contact   : allqoow@gmail.com
# Started on: 20161107(yyyymmdd)
# Project   : exerciseML(Exercise for Machine Learning)	

import matplotlib.pyplot
import math
import numpy.random


setX = [-1,0.3,2]
setT = [-0.1,0.5,0.5]

def getGradient(setX,setT,vecW):

	ret