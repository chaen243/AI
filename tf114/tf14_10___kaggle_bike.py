import tensorflow as tf
import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())


x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 53, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
tp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1], dtype = tf.float32, name= 'weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype = tf.float32, name= 'bias'))

#2. 모델

hypothesis = tf.compat.v1.matmul(xp,w)+b

#3-1. 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - yp)) #mse

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.005)
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
