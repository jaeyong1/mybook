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

# Build a graph.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.multiply(a, b)
print("node c = ", c)

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print("run c = ", sess.run(c, feed_dict={a: 3., b: 4.}))

 