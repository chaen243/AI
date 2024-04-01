

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

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42, stratify=y)

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 64])
w = tf.compat.v1.Variable(tf.random_normal([64,10]))
b = tf.compat.v1.Variable(tf.zeros([1,10]))
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#2. 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis =1 ))

train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate= 0.5).minimize(loss)

from sklearn.metrics import r2_score, accuracy_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 100001
for step in range(epochs):
    cost_val, _ , w_val= sess.run([loss, train, w],
                           feed_dict = {x : x_train, y : y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
   
   
#4. 평가. 예측        
y_predict = sess.run(hypothesis, feed_dict= {x : x_test})   
r2 = r2_score(y_test, y_predict) 

print('y_pred :', y_predict)
print('r2 :', r2)    
print(w_val)

y_predict = np.argmax(y_predict,1)
print(y_predict)        

acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

#로스가 너무 커...
