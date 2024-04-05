import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split




#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255 #(0~1사이 정규화)
#통상 2가지 방법사용
#x_train = x_train.reshape(60000, 28*28).astype('float32')/127.5 (-1~1사이 정규화)

x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 실습

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 784])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)

#가중치

# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784,512], name= 'weights')) 

w1 = tf.compat.v1.get_variable('w1', shape=[784, 512],)
                               #initializer=tf.contrib.layers.xavier_initializer()) #= random_normal 디폴트
#가중치 최초 초기값을 주고 시작점을 잘 잡아주고 initializer는 1에포에서만 사용되게 만들어져있음. 2에포에서는 적용되지 않게 만들어져있음.

b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([512], name = 'bias'))
layer1 = tf.compat.v1.matmul(x, w1) + b1 
layer1 =  tf.compat.v1.nn.relu(tf.compat.v1.nn.dropout(layer1, rate = 0.3))


w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([512,256], name= 'weights')) 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([256], name = 'bias'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2
layer2 = tf.compat.v1.nn.relu(tf.compat.v1.nn.dropout(layer2, rate = 0.3))


w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([256,256], name= 'weights')) 
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([256], name = 'bias'))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3
layer3 = tf.compat.v1.nn.relu(layer3)


w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([256,128], name= 'weights')) 
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([128], name = 'bias'))
layer4 = tf.compat.v1.matmul(layer3, w4) + b4
layer4 = tf.compat.v1.nn.relu(layer4)


w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128,64], name= 'weights')) 
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))
layer5 = tf.compat.v1.matmul(layer4, w5) + b5
layer5 = tf.compat.v1.nn.relu(layer5)


w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name= 'weights')) 
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))
layer6 = tf.compat.v1.matmul(layer5, w6) + b6
layer6 = tf.compat.v1.nn.softmax(layer6)


w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,10], name= 'weights')) 
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'bias'))
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer6, w6) + b6)

#3-1. 컴파일

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
#동일!
#loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.nn.log_softmax(hypothesis), axis=1))

  

train = tf.compat.v1.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)


#4. 평가. 예측     
from sklearn.metrics import r2_score, accuracy_score
  

'''
epochs = 1001
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(epochs):
        cost_val, _ = sess.run([loss, train], feed_dict={x: x_train, y: y_train, keep_prob: 0.2})
        if step % 100 == 0:
            print(step, 'loss:', cost_val)
    
    hypo = sess.run(hypothesis, feed_dict={x: x_test, keep_prob: 1.0})
    y_predict = tf.argmax(hypo, 1)
    y_test = tf.argmax(y_test, 1)
    accuracy_val = tf.reduce_mean(tf.cast(tf.equal(y_predict, y_test), dtype=tf.float32))
    
    acc = sess.run(accuracy_val)
    print('정확도:', acc)
'''  

#선생님 version
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4],
                                          feed_dict = {x: x_train, y: y_train})
                                          
    if step % 20 == 0:
        print(step, 'loss :', cost_val )
        
#4. 평가
y_predict = sess.run(hypothesis, feed_dict= {x: x_test})                                              
# print(y_predict[0], y_predict.shape) # (10000, 10)
y_predict_arg = sess.run(tf.math.argmax(y_predict, 1))
y_test_arg = sess.run(tf.math.argmax(y_test, 1))

#print(y_predict_arg[0], y_predict_arg.shape)

acc = accuracy_score(y_predict_arg, y_test_arg)
print('acc :', acc)


        

