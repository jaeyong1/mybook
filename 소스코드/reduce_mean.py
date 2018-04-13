# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:54:13 2018

@author: JY


"""
# [참고자료] reduce_mean()
import tensorflow as tf
x = tf.constant([[1., 1.], [2., 2.]])

sess= tf.Session()

print("Rank=", sess.run(
        tf.rank(x) #Rank=2
        ))

print(sess.run(
        tf.reduce_mean(x) # 1.5
        ))  

print(sess.run(
        tf.reduce_mean(x, axis=0) # [1.5, 1.5]
        )) 

print(sess.run(
        tf.reduce_mean(x, axis=1) # [1.,  2.]
        )) 

'''
#실행결과
Rank= 2
1.5
[ 1.5  1.5]
[ 1.  2.]
'''