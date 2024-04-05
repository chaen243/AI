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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 282 ) #282

y_train = np.expand_dims(y_train, axis=1)
y_testn = np.expand_dims(y_test, axis=1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.compat.v1.placeholder(tf.float32)




w1 = tf.compat.v1.Variable(tf.random_normal([8,512]))
b1 = tf.compat.v1.Variable(tf.zeros([512]))
layer1 = tf.nn.relu(tf.compat.v1.matmul(xp, w1) + b1) 
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)


w2 = tf.compat.v1.Variable(tf.random_normal([512,256]))
b2 = tf.compat.v1.Variable(tf.zeros([256]))
layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1, w2) + b2) 

w3 = tf.compat.v1.Variable(tf.random_normal([256,128]))
b3 = tf.compat.v1.Variable(tf.zeros([128]))
layer3 = tf.nn.relu(tf.compat.v1.matmul(layer2, w3) + b3) 


w4 = tf.compat.v1.Variable(tf.random_normal([128,64]))
b4 = tf.compat.v1.Variable(tf.zeros([64]))
layer4 = tf.nn.relu(tf.compat.v1.matmul(layer3, w4) + b4) 

w5 = tf.compat.v1.Variable(tf.random_normal([64,32]))
b5 = tf.compat.v1.Variable(tf.zeros([32]))
layer5 = tf.nn.relu(tf.compat.v1.matmul(layer4, w5) + b5)


w6 = tf.compat.v1.Variable(tf.random_normal([32,16]))
b6 = tf.compat.v1.Variable(tf.zeros([16]))
layer6 = tf.nn.relu(tf.compat.v1.matmul(layer5, w6) + b6 )

w7 = tf.compat.v1.Variable(tf.random_normal([16,1]))
b7 = tf.compat.v1.Variable(tf.zeros([1]))
hypothesis = tf.compat.v1.matmul(layer6, w7) + b7 
#2. 모델


#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - yp))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.15)
train = optimizer.minimize(loss)

#3-2. 훈련
from sklearn.metrics import r2_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],
                           feed_dict= {xp : x_train, yp : y_train, keep_prob : 0.8})
    if step % 50 ==0:
        print(step, 'loss:', cost_val)
        
y_predict = sess.run(hypothesis, feed_dict= {xp : x_test, keep_prob : 1.0})   
r2 = r2_score(y_test, y_predict) 

print('y_pred :', y_predict)
print('r2 :', r2)    

