#드랍아웃 적용
#훈련/평가를 분리하지 않은 상태.
#tensor 2.x에는 적용되어있음.
#tensor1.x에서는 parameter 줄때 placeholder로 넣어주면 됨


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

keep_prob = tf.placeholder(tf.float32)

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle= True, stratify= y, random_state=206)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 1])





#2. 모델
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,128], name= 'weights')) 
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128], name = 'bias'))
layer1 = tf.compat.v1.matmul(x, w1) + b1 
later1 = tf.compat.v1.nn.dropout(layer1, keep_prob=keep_prob)


w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128,64], name= 'weights')) 
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([64], name = 'bias'))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2


w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name= 'weights')) 
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], name = 'bias'))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,8], name= 'weights')) 
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([8], name = 'bias'))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1], name= 'weights')) 
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias'))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

#3-1. 컴파일

loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1 -y)*tf.log(1-hypothesis))
    #binary_crossentropy

optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

#4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype= tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(8001):
        cost_val, _ = sess.run([loss, train], feed_dict= {x: x_train, y: y_train, keep_prob:0.5}) #훈련은 dropout 적용
        
        if step % 200 == 0:
            print(step, cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, acc],
                               feed_dict = {x:x_test, y:y_test, keep_prob:1.0}) #predict/acc는 dropout 미적용
    print('훈련값 :', hypo)
    print('예측값 :', pred)
    print('정확도 :', acc)
        

# 정확도 : 0.8245614        

#dropout
#정확도 : 0.88596493