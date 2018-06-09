# -*- coding: utf-8 -*-
"""
비행기무게에 따른 이륙거리 예측하기

Created on Fri Apr 20 2018

@author: jynote
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#데이터셋을 섞음
def shuffle_data(x_train,y_train):
  temp_index = np.arange(len(x_train))

  #Random suffle index
  np.random.shuffle(temp_index)

  #Re-arrange x and y data with random shuffle index
  x_temp = np.zeros(x_train.shape)
  y_temp = np.zeros(y_train.shape)
  x_temp = x_train[temp_index]
  y_temp = y_train[temp_index]        

  return x_temp, y_temp

#0에서 1사이 값으로 정규화
def minmax_normalize(x):
  xmax, xmin = x.max(), x.min()
  return (x - xmin) / (xmax - xmin)

#정규화 된 값을 real값으로 변환
def minmax_get_norm(realx, arrx):
  xmax, xmin = arrx.max(), arrx.min()  
  normx = (realx - xmin) / (xmax - xmin)
  return normx    
    
#0에서 1사이 값을 실제 값으로 역정규값 리턴
def minmax_get_denorm(normx, arrx):
  xmax, xmin = arrx.max(), arrx.min()
  realx = normx * (xmax - xmin) + xmin
  return realx
  

def main():
  traincsvdata = np.loadtxt('trainset.csv', unpack=True, delimiter=',', skiprows=1)   
  num_points = len(traincsvdata[0]) 
  print("training points : ", num_points)

  x_data = traincsvdata[0]  # [230. 280. 241....
  y_data = traincsvdata[1]  # [1349.9 1809.  1590.8 1571.8 1768.3

  #학습용 데이터셋을 녹색(g)에 둥근점(o)로 시각화
  plt.suptitle('Training Data Set', fontsize=16)
  plt.plot(x_data, y_data, 'go')
  plt.xlabel('weight')
  plt.ylabel('distance')
  plt.show()

  #데이터 정규화 진행. 데이터범위를 0~1사이 값으로 변환.
  x_data = minmax_normalize(x_data)
  y_data = minmax_normalize(y_data)
  
  #배치단위로 학습
  BATCH_SIZE = 5
  BATCH_NUM = int(len(x_data)/BATCH_SIZE)
  
  #데이터를 세로로(한개씩)나열한 형태로 reshape
  x_data = np.reshape(x_data, [len(x_data),1]) # [[230.] [280.] ...
  y_data = np.reshape(y_data, [len(y_data),1])
  
  #총 개수는 정해지지 않았고 1개씩 들어가는 Placeholder 생성
  input_data = tf.placeholder(tf.float32, shape=[None,1])  
  output_data = tf.placeholder(tf.float32, shape=[None,1])

  #레이어간 Weight 정의후 랜덤값으로 초기화. 그림에서는 선으로 표시.
  W1 = tf.Variable(tf.random_uniform([1,5], 0.0, 300.0))
  W2 = tf.Variable(tf.random_uniform([5,3], 0.0, 1.0))
  W_out = tf.Variable(tf.random_uniform([3,1], 0.0, 300.0))

  #레이어의 노드가 하는 계산. 이전노드와 현재노드의 곱셈. 비선형함수로 sigmoid 추가.
  hidden1 = tf.nn.sigmoid(tf.matmul(input_data,W1))
  hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,W2))
  output = tf.matmul(hidden2, W_out)

  #비용함수, 최적화함수, train 정의
  loss = tf.reduce_mean(tf.square(output-output_data))
  optimizer = tf.train.AdamOptimizer(0.1)
  train = optimizer.minimize(loss)

  #변수(Variable) 사용준비
  init = tf.global_variables_initializer()

  #세션 열고 init 실행
  sess= tf.Session()
  sess.run(init)


  #학습을 반복하며 값 업데이트
  for step in range(4000):
    index = 0    

    #매번 데이터셋을 섞음
    x_data, y_data = shuffle_data(x_data, y_data)

    #배치크기만큼 학습을 진행
    for batch_iter in range(BATCH_NUM-1):
        feed_dict = {input_data: x_data[index:index+BATCH_SIZE], output_data: y_data[index:index+BATCH_SIZE]}
        sess.run(train, feed_dict = feed_dict)
        index += BATCH_SIZE
        
    #화면에 학습진행상태 출력(최초100회까지는 10마다 한번씩, 이후는 100회에 한번씩)
    if (step%200==0): 
        print("Step=%5d, Loss Value=%f" %(step, sess.run(loss, feed_dict = feed_dict)))      

  
  #학습이 끝난후 그래프로 결과확인
  print("## 학습결과 그래프 ##")
  
  #학습용 데이터셋을 녹색(g)에 둥근점(o)로 시각화    
  plt.plot(x_data, y_data, 'go')  
  
  #예측모델 출력은 검은색(k) 별표(*)로 시각화
  feed_dict = {input_data: x_data}
  plt.plot(x_data, sess.run(output, feed_dict=feed_dict), 'k*') 
  
  #그래프 그리기
  plt.suptitle('Training Result', fontsize=16)    
  plt.xlabel('weight')
  plt.ylabel('distance')
  plt.show()
  
  print("# 학습결과 이륙거리계산 #")
  ask_x = 270
  ask_norm_x = [[minmax_get_norm(ask_x, traincsvdata[0])]]
  answer_norm_y = sess.run(output, feed_dict={input_data: ask_norm_x})
  answer_y = minmax_get_denorm(answer_norm_y, traincsvdata[1])
  print("> 무게(X):", ask_x, "ton => 이륙거리Y:", answer_y[0][0], "m \n\n\n")
  
  #테스트셋을 활용한 결과확인  
  print("## Test Set 검증결과 그래프 ##")
        
  #테스트셋 파일읽음
  test_csv_x_data = np.loadtxt('testset_x.csv', unpack=True, delimiter=',', skiprows=1)
  test_csv_y_data = np.loadtxt('testset_y.csv', unpack=True, delimiter=',', skiprows=1)
  
  #테스트셋 정규화 진행
  test_norm_x_data = minmax_normalize(test_csv_x_data)
  test_norm_y_data = minmax_normalize(test_csv_y_data)
  
  #CVS 데이터(테스트셋, 정답데이터) : 빨간색(m)에 둥근점(o)로 시각화
  plt.plot(test_csv_x_data, test_csv_y_data, 'mo')
  
  #예측데이터 : 검은색(k) 별표(*)로 시각화   
  feed_dict = {input_data: np.reshape(test_norm_x_data, (len(test_norm_x_data), 1)) } #[[0.41573034] [0.20224719] ...
  test_pred_y_data = minmax_get_denorm(sess.run(output, feed_dict=feed_dict), traincsvdata[1])
  plt.plot(test_csv_x_data, test_pred_y_data, 'k*')
  
  #그래프 그리기
  plt.suptitle('Test Result', fontsize=16)  
  plt.xlabel('weight')
  plt.ylabel('distance')
  plt.show()

#main함수
if __name__ == "__main__":
  main()
