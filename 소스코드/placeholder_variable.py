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


input_data = [1,2,3,4,5]
x = tf.placeholder(dtype=tf.float32)
W = tf.Variable([2],dtype=tf.float32)
y = W*x


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict={x:input_data})


print(result)
