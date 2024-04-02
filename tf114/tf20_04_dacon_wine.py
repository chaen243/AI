#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time



#1. 데이터
path = "C:\\_data\\daicon\\wine\\"



train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
print(x)
y = train_csv['quality']-3
y = np.array(y)

print(x.shape, y.shape) #(5497, 12) (5497,)
#print(pd.value_counts(y))



x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=67, shuffle=True, stratify= y) #67


x = tf.compat.v1.placeholder(tf.float32, shape= [None, 12])
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 7])


w1 = tf.compat.v1.Variable(tf.random_normal([12,256]))
b1 = tf.compat.v1.Variable(tf.zeros([256]))
layer1 = tf.compat.v1.matmul(x, w1)+ b1



w2 = tf.compat.v1.Variable(tf.random_normal([256,128]))
b2 = tf.compat.v1.Variable(tf.zeros([128]))
layer2 = tf.compat.v1.matmul(layer1, w2)+ b2


w3 = tf.compat.v1.Variable(tf.random_normal([128,64]))
b3 = tf.compat.v1.Variable(tf.zeros([64]))
layer3 = tf.compat.v1.matmul(layer2, w3)+ b3


w4 = tf.compat.v1.Variable(tf.random_normal([64,32]))
b4 = tf.compat.v1.Variable(tf.zeros([32]))
layer4 = tf.nn.softmax(tf.compat.v1.matmul(layer3, w4)+ b4)


w5 = tf.compat.v1.Variable(tf.random_normal([32,7]))
b5 = tf.compat.v1.Variable(tf.zeros([7]))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5)+ b5)



scaler = StandardScaler()
#scaler = MinMaxScaler()
 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델


#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis =1 ))

train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate= 0.005).minimize(loss)

from sklearn.metrics import r2_score, accuracy_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 3001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train ],
                           feed_dict = {x : x_train, y : y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
   
   
#4. 평가. 예측        
y_predict = sess.run(hypothesis, feed_dict= {x : x_test})   

print('y_pred :', y_predict)

y_predict = np.argmax(y_predict,1)
y_test = np.argmax(y_test,1)
print(y_predict)        

acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

# acc :  0.16363636363636364


