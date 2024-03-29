#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"
# print(path + "aaa.csv") 문자그대로 보여줌 c:\_data\daicon\ddarung\aaa.csv
# pandas에서 1차원- Series, 2차원이상은 DataFrame이라고 함.

train_csv = pd.read_csv(path + "train.csv", index_col=['id']) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
#submission_csv = pd.read_csv(path + "submission.csv") 
#print(submission_csv)


###########################결측치처리########################


train_csv = train_csv.fillna(train_csv.ffill())  #뒤 데이터로 채움
print(train_csv.isnull().sum())
test_csv = test_csv.fillna(test_csv.ffill())  #앞 데이터로 채움
print(test_csv.isnull().sum())

print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)      #(1328, 10)
print(test_csv.info()) # 717 non-null

###########################결측치처리########################




################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
#print(x)
y = train_csv['count']
#print(y)


print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= False, random_state= 17068) #399 #1048 #6

#scaler = MinMaxScaler()
# scaler = StandardScaler()


# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)


y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 9])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9, 1], dtype = tf.float32, name= 'weights'))
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