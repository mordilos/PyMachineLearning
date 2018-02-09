#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:08:08 2018

@author: neck
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Dense(1, input_dim=500))
model.add(Activation(activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

data = np.random.random((1000, 500))

labels = np.random.randint(2, size=(1000, 1))

score = model.evaluate(data,labels, verbose=0)

print "Before Training:", zip(model.metrics_names, score)

model.fit(data, labels, nb_epoch=10, batch_size=32, verbose=0)

score = model.evaluate(data,labels, verbose=0)

print "After Training:", zip(model.metrics_names, score)

plot_model(model, to_file='s1.png', show_shapes=True)