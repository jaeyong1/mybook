# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:54:13 2018

@author: JY
"""

from __future__ import division
from __future__ import print_function
import os.path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



#MNIST를 인터넷에서 받아서 mnist_data 폴더에 저장
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

    
#화면에 표시함수
def print_info(x):
    print('shape : %s \n' % (x.shape,))
    print('value \n%s' % (x) )

print("=train set=")    
print_info(mnist.train.images)
print_info(mnist.train.labels)

print("=test set=")    
print_info(mnist.test.images)
print_info(mnist.test.labels)

print("=validation set=")   
print_info(mnist.validation.images)
print_info(mnist.validation.labels)

#show 6 images from random index
for i in np.random.randint(55000, size=6):
    imgvec = mnist.train.images[i, :]
    labelvec = mnist.train.labels[i, :]
    imagematrix = np.reshape(imgvec, (28, 28)) # (784,) -> (28, 28)
    label = np.argmax(labelvec) #[0 0 1..] -> 2    
    plt.matshow(imagematrix, cmap=plt.get_cmap('gray'))
    plt.title("Index: %d, Label: %d" % (i, label))
    
    
#RANDOM minibatch
