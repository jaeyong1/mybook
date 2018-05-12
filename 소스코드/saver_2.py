'''
복구 하기
'''
import tensorflow as tf

b1= tf.Variable(0.0, name="bias")
  
# Saver 생성
saver = tf.train.Saver()
  
# 세션준비. init을 하지 않습니다. 
sess= tf.Session()

# 파일이 있으면 Variable 복구
ckpt = tf.train.get_checkpoint_state('./saver_bias/')
print(ckpt)
if tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
  saver.restore(sess, ckpt.model_checkpoint_path)
  print("variable is restored")

# 복구된 내용 확인
print("bias:", sess.run(b1))
