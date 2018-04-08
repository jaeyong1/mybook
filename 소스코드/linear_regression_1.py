# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:54:13 2018

@author: JY


from __future__ import division
from __future__ import print_function
import os.path

import numpy as np
import matplotlib.pyplot as plt

"""

import tensorflow as tf

# X and Y data
x_train = [1, 2, 3, 4]
y_train = [6, 5, 7, 10]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
   sess.run(train)
   print(step, sess.run(cost), sess.run(W), sess.run(b))


#   if step % 20 == 0:
#       print(step, sess.run(cost), sess.run(W), sess.run(b))
