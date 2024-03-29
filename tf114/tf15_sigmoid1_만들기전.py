import tensorflow as tf
tf.compat.v1.set_random_seed(777)


#1. 데이터 

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] #(6,1)

################################################################################
#그냥해보기
################################################################################

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name = 'weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))

#2. 모델

hypothesis = tf.compat.v1.matmul(x,w)+b

#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.000005)
train = optimizer.minimize(loss)

#3-2. 훈련

loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                            feed_dict = {x : x_data, y : y_data})
    if step % 20 ==0:
        print(step, "loss:", cost_val, '\t', w_val)
        
        loss_history.append(cost_val)
        
#4. 평가, 예측        
y_predict = sess.run(hypothesis, feed_dict={x : x_data})

# y_pred = x * w_val + b_val #안먹힘!

###y_predict랑 같음
x_test = tf.compat.v1.placeholder(tf.float32, shape= [None, 2])
y_pred = tf.matmul(x_test, w_val) + b_val
y_predd = sess.run(y_pred, feed_dict= {x_test : x_data})
print(w_val) #[[0.7927159 ]  [0.66863364]]
print(type(w_val)) #<class 'numpy.ndarray'> #tensorflow의 데이터 형태!
print('y_pred :', y_predict) 
print('y_predd :', y_predd) 

# y_pred : [[2.120012 ]
#  [3.5813615]
#  [3.0368102]
#  [5.1667933]
#  [5.9595094]
#  [6.0835915]]

sess.close()       