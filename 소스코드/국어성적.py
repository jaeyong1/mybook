import tensorflow as tf

#x_train = [1,2,3]
#y_train = [1,2,3]
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis = x_train * W + b
hypothesis = X * W + b

#cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5001):
    #sess.run(train)
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train], feed_dict={X:[5, 7], Y:[52, 72]})
    if step%500 == 0:
        print(step, cost_val, W_val, b_val)
#        print(step, sess.run(cost), sess.run(W), sess.run(b))        
#        plt.plot(x_train, y_train, 'ro')
#        plt.plot(x_train, sess.run(W) * x_train + sess.run(b))
#        plt.xlabel('x_train')
#        plt.ylabel('y_train')
#        plt.show()
#        
        
#ask question, give x, then guess y
print("예측Y : ", sess.run(hypothesis, feed_dict={X:[8]}))




        