# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:50:07 2017

https://www.kaggle.com/c/titanic/data
Titanic: Machine Learning from Disaster

@author: JY
"""
#   'names':   ('PassengerId',     'Survived',  'Pclass',   'Name',  'Sex',      'Age',      'SibSp',    'Parch',   'Ticket', 'Fare'    , 'Cabin', 'Embarked'),                
  #dtype={
  #              'names': ('PassengerId',     'Survived',  'Pclass',   'Name',  'Sex'),
  #              'formats': ( np.int, np.float, np.float,  np.str,  np.str)
  #              },

import numpy as np
import pandas
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import cv2
#import pyscreenshot as ImageGrab #[JYP] NOT WORKING T-T
from PIL import ImageGrab
import time

#Load CSV file as matrix
csvdata = pandas.read_csv('train.csv').as_matrix()
csvdata2 = pandas.read_csv('test.csv').as_matrix()
csvdata3 = pandas.read_csv('gender_submission.csv').as_matrix()

#MALE -> 1
#FEMALE -> 0
for i in range(len(csvdata)):
    #print(csvdata[i , 4])
    if csvdata[i , 4] == 'male':
        csvdata[i, 4] = 1
    else:
        csvdata[i, 4] = 0

for i in range(len(csvdata2)):
    #print(csvdata[i , 4])
    if csvdata2[i , 3] == 'male':
        csvdata2[i, 3] = 1
    else:
        csvdata2[i, 3] = 0



#Age Black -> fill average
#get index that is not NAN as age
index_not_nan = np.where(np.logical_not(np.isnan(csvdata[:,5].astype(float))))
avg_age = np.average(csvdata[index_not_nan, 5].astype(float))
print("평균나이 : " , avg_age)
#set age to NAN 
for i in range(len(csvdata)):
    if np.isnan(csvdata[i , 5]):
        csvdata[i, 5] = avg_age
print("평균나이2 : " , np.average(csvdata[:,5]))


for i in range(len(csvdata2)):
    if np.isnan(csvdata2[i , 4]):
        csvdata2[i, 4] = avg_age



#Embarked
# Empty -> 0
# S -> 1
# C -> 2
# Q -> 3
for i in range(len(csvdata)):        
    if csvdata[i , 11] == 'S':
        csvdata[i, 11] = 1
    elif csvdata[i , 11] == 'C':
        csvdata[i, 11] = 2
    elif csvdata[i , 11] == 'Q':
        csvdata[i, 11] = 3    
    if np.isnan(csvdata[i, 11]):
        csvdata[i, 11] = 0               

for i in range(len(csvdata2)):        
    if csvdata2[i , 10] == 'S':
        csvdata2[i, 10] = 1
    elif csvdata2[i , 10] == 'C':
        csvdata2[i, 10] = 2
    elif csvdata2[i , 10] == 'Q':
        csvdata2[i, 10] = 3    
    if np.isnan(csvdata2[i, 10]):
        csvdata2[i, 10] = 0               


#Fare
print("avg Fare : ", np.average(csvdata[:, 9].astype(float)))

X_PassengerData = csvdata[:, [2, #Pclass
                           4, #Sex
                           #5, #Age 
                           6, #SibSp
                           7#, #Parch
                          #, 9#, #Fare
                          , 11 #Embarked
                           ] ]
Y_Survived = csvdata[:, 1:2]

Test_X_PassengerData = csvdata2[:, [1, #Pclass
                           3, #Sex
                           #4, #Age 
                           5, #SibSp
                           6#, #Parch
                          #, 8#, #Fare
                          , 10 #Embarked
                           ] ]
Test_Y_Survived = csvdata3[:, 1:2]

print(X_PassengerData)
#print(Y_Survived)

#placeholder
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis = x_train * W + b
#hypothesis = tf.matmul(X,W) + b
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)


#cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

#Minimize
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)
#train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

previous_cost = 0
#Lauch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10000):#10000
#       cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: X_PassengerData, Y:Y_Survived})
        cost_val,  hy_val, _ =sess.run([cost, hypothesis, train], feed_dict={X:X_PassengerData, Y:Y_Survived})
    
        if step%800 == 0:
            print(step, "\nCost: ", cost_val, "  \nPrediction : ", hy_val)
           
        #cost 진척이 없는데 굳이 반복 필요없음
        if previous_cost == cost_val:
           print("found best hyphothesis when step ", step)
           break
        else:
           previous_cost = cost_val
    
    #가설검증(설명력)
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:X_PassengerData, Y:Y_Survived})
    print("\nHypothesis: ", h , "\nCorrect(Y): " , c, "\nAccuracy: " , a)
    print("\nTest CSV runningResult\n")
    h2,c2,a2 = sess.run([hypothesis, predicted, accuracy], feed_dict={X:Test_X_PassengerData, Y:Test_Y_Survived})
    print("\nHypothesis: ", h2 , "\nCorrect(Y): " , c2, "\nAccuracy: " , a2)
    #for h in range(len(csvdata2)):
    #    print(c2[h,0])

print("end~")
