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

import numpy as np
import tensorflow as tf 

# Build a graph.
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

# Launch the graph in a session.
sess=tf.Session()

# Evaluate the tensor `y`.
#print(sess.run(y)) # ERROR: will fail because x was not fed.

rand_array = np.random.rand(1024, 1024)
print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
