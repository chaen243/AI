import tensorflow as tf

tf.compat.v1.set_random_seed(287)

#1. 데이터

x1_data = [73., 93., 89., 96., 73.] #국어
x2_data = [80., 88., 91., 98., 66.] #영어
x3_data = [75., 93., 90., 100., 70.] #수학
y_data = [152., 185., 180., 196., 142.] #환산

x1 = tf.compat.v1.placeholder(tf.float32, shape= [None])
x2 = tf.compat.v1.placeholder(tf.float32, shape= [None])
x3 = tf.compat.v1.placeholder(tf.float32, shape= [None])
y = tf.compat.v1.placeholder(tf.float32, shape= [None])

w_1 = tf.compat.v1.Variable([1], dtype=tf.float32, name= 'weight1')
w_2 = tf.compat.v1.Variable([1], dtype=tf.float32, name= 'weight2')
w_3 = tf.compat.v1.Variable([1], dtype=tf.float32, name= 'weight3')
b = tf.compat.v1.Variable([1], dtype=tf.float32, name= 'bias')

#2.모델
hypothesis = x1_data * w_1 + x2_data * w_2 + x3_data * w_3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)





#3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(2001):
    cost_val, _ =sess.run([loss, train],
                          feed_dict= {x1 : x1_data, x2 : x2_data, x3 : x3_data, y : y_data})
    if step % 20 == 0:
        print(step, "loss:", cost_val, )

    w_history.append(w_v)
    loss_history.append(cost_val)
    
y_predict = x_test * w_v    
mae_loss = mean_absolute_error(y_test, y_predict)    
r2 = r2_score(y_test,y_predict)

print('예측값 :', y_predict)
print('mae : ',mae_loss)
print('r2 : ', r2)

sess.close()