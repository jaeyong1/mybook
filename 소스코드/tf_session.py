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
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print("c = ", c)

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print("sess.run(c) = ", sess.run(c))

# 실행결과
#c =  Tensor("mul:0", shape=(), dtype=float32)
#sess.run(c) =  30.0