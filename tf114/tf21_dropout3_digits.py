

from sklearn.datasets import load_digits
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import time


datasets = load_digits() #mnist 원판
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape) #(1797, 64)
print(y.shape) #(1797,)
print(pd.value_counts(y, sort= False)) #sort= False 제일 앞 데이터부터 순서대로 나옴
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse= False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=947, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape= [None, 64])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random_normal([64,512]))
b1 = tf.compat.v1.Variable(tf.zeros([512]))
layer1 = tf.nn.relu(tf.compat.v1.matmul(x, w1) + b1) 
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
layer5 = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5) 


w6 = tf.compat.v1.Variable(tf.random_normal([32,10]))
b6 = tf.compat.v1.Variable(tf.zeros([10]))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer5, w6) + b6) 



#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))


train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate= 0.4).minimize(loss)

from sklearn.metrics import r2_score, accuracy_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5001
for step in range(epochs):
    cost_val, _ , = sess.run([loss, train],
                           feed_dict = {x : x_train, y : y_train, keep_prob : 0.8})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
   
   
#4. 평가. 예측        
y_predict = sess.run(hypothesis, feed_dict= {x : x_test, keep_prob : 1.0})   
r2 = r2_score(y_test, y_predict) 

print('y_pred :', y_predict)

y_predict = np.argmax(y_predict,1)
y_test = np.argmax(y_test,1)

print('r2 :', r2)


# r2 : 0.020781581119223937

#dropout 적용
# r2 : 0.1550031810246936