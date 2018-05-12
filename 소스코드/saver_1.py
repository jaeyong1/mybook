'''
저장 하기
'''
import tensorflow as tf

#그래프에서 사용하는 변수
b1= tf.Variable(2.0, name="bias") #save test
  
#변수사용 준비
init = tf.global_variables_initializer()

# Saver 생성
saver = tf.train.Saver()
  
# 세션준비
sess= tf.Session()
sess.run(init)

# 학습과정을 간단히 print로 대체
print("save test bias", sess.run(b1))

# 디스크에 변수를 저장
save_path = saver.save(sess, "./saver_bias/bias.ckpt")
print("Model saved in file: %s" % save_path)
