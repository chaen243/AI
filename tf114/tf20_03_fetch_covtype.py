import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score


#1. 데이터


datasets = fetch_covtype()
print(datasets)
print(datasets.DESCR)

print(datasets.feature_names) #'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
#'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 
# 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 
# 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 
# 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20',
# 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29',
# 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38',
# 'Soil_Type_39']

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(581012, 54) (581012,)
print(pd.value_counts(y))
print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)

#y = pd.get_dummies(y)

y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  shuffle= True, random_state= 123, stratify= y) #y의 라벨값을 비율대로 잡아줌 #회귀모델에서는 ㄴㄴ 분류에서만 가능
#print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
#print(y_train.shape, y_test.shape) #(7620, ) (3266, )
print(np.unique(y_test, return_counts = True ))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xp = tf.compat.v1.placeholder(tf.float32, shape= [None, 54])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])


#2. 모델구성
w1 = tf.compat.v1.Variable(tf.random_normal([54,512]))
b1 = tf.compat.v1.Variable(tf.zeros([512]))
layer1 = tf.compat.v1.matmul(xp, w1) + b1 


w2 = tf.compat.v1.Variable(tf.random_normal([512,256]))
b2 = tf.compat.v1.Variable(tf.zeros([256]))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2


w3 = tf.compat.v1.Variable(tf.random_normal([256,128]))
b3 = tf.compat.v1.Variable(tf.zeros([128]))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3


w4 = tf.compat.v1.Variable(tf.random_normal([128,64]))
b4 = tf.compat.v1.Variable(tf.zeros([64]))
layer4 = tf.compat.v1.matmul(layer3, w4) + b4


w5 = tf.compat.v1.Variable(tf.random_normal([64,32]))
b5 = tf.compat.v1.Variable(tf.zeros([32]))
layer5 = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5) 
 

w6 = tf.compat.v1.Variable(tf.random_normal([32,7]))
b6 = tf.compat.v1.Variable(tf.zeros([7]))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer5, w6) + b6) 
 
 
 





#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(hypothesis), axis =1 ))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(loss)

from sklearn.metrics import r2_score, accuracy_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 201
for step in range(epochs):
    cost_val, _ , w_val= sess.run([loss, train, w6],
                           feed_dict = {xp : x_train, yp : y_train})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)
   
   
#4. 평가. 예측        
y_predict = sess.run(hypothesis, feed_dict= {xp : x_test})   

print('y_pred :', y_predict)
print(w_val)

y_predict = np.argmax(y_predict,1)
y_test = np.argmax(y_test,1)

#print(y_predict)        

acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

# acc :  0.5021643158954588

