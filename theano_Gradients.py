#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:44:11 2018

@author: neck
"""

import theano.tensor as T
from theano import function
from theano import shared
import numpy

x = T.dmatrix('x')

y = shared(numpy.array([[4, 5, 6]]))

z = T.sum(((x * x) + y) * x)

f = function(inputs = [x], outputs = [z])

g = T.grad(z,[x])

g_f = function([x], g)

print "Original:", f([[1, 2, 3]])
print "Original Gradient:", g_f([[1, 2, 3]])

y.set_value(numpy.array([[1, 1, 1]]))

print "Updated:", f([[1, 2, 3]])
print "Updated Gradient", g_f([[1, 2, 3]])