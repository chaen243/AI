import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
tf.compat.v1.set_random_seed(777)
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 3884 ) #282

y_train = np.expand_dims(y_train, axis=1)
y_testn = np.expand_dims(y_test, axis=1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1], dtype = tf.float32, name = 'weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype = tf.float32, name = 'bias'))

#2. 모델

hypothesis = tf.compat.v1.matmul(xp,w)+b

#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - yp))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005)
train = optimizer.minimize(loss)

#3-2. 훈련
from sklearn.metrics import r2_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],
                           feed_dict= {xp : x_train, yp : y_train})
    if step % 20 ==0:
        print(step, 'loss:', cost_val)
        
y_predict = sess.run(hypothesis, feed_dict= {xp : x_test})   
r2 = r2_score(y_test, y_predict) 

print('y_pred :', y_predict)
print('r2 :', r2)    

