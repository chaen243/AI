import os
import random as rn
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
le = LabelEncoder()
import time



#1. 데이터

path = 'C:\\_data\\kaggle\\Obesity_Risk\\'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#print(train_csv)
#print(test_csv)
#print(submission_csv)

# print(train_csv.shape) #(20758, 17)
# print(test_csv.shape) #(13840, 16)
# print(submission_csv.shape) #(13840, 2)

#print(train_csv.columns) #Index(['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#       'CALC', 'MTRANS', 'NObeyesdad'],



#print(train_csv['NObeyesdad'].value_counts())
# Obesity_Type_III       4046
# Obesity_Type_II        3248
# Normal_Weight          3082
# Obesity_Type_I         2910
# Insufficient_Weight    2523
# Overweight_Level_II    2522
# Overweight_Level_I     2427


##############데이터 전처리###############
le = LabelEncoder()



train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
#train_csv['WIR'] = train_csv['Weight'] / train_csv['CH2O']
train_csv['bmioncp'] = train_csv['BMI'] / train_csv['NCP']


test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
#test_csv['WIR'] = test_csv['Weight'] / test_csv['CH2O']
test_csv['bmioncp'] = test_csv['BMI'] / test_csv['NCP']

#Gender
train_csv['Gender']= train_csv['Gender'].str.replace("Male","0")
train_csv['Gender']= train_csv['Gender'].str.replace("Female","1")
test_csv['Gender']= test_csv['Gender'].str.replace("Male","0")
test_csv['Gender']= test_csv['Gender'].str.replace("Female","1")

# print(train_csv['Gender'])
# print(test_csv['Gender'])



#family_history_with_overweight
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("yes","0")
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("no","1")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("yes","0")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("no","1")

# print(train_csv['family_history_with_overweight'])
# print(test_csv['family_history_with_overweight'])

train_csv['FAVC']= train_csv['FAVC'].str.replace("yes","0")
train_csv['FAVC']= train_csv['FAVC'].str.replace("no","1")
test_csv['FAVC']= test_csv['FAVC'].str.replace("yes","0")
test_csv['FAVC']= test_csv['FAVC'].str.replace("no","1")

#print(train_csv['FAVC'])
#print(test_csv['FAVC'])
#print(np.unique(train_csv['FAVC'], return_counts= True))
#print(np.unique(test_csv['FAVC'], return_counts= True))


#print(np.unique(train_csv['CAEC'], return_counts= True))
train_csv['CAEC']= train_csv['CAEC'].str.replace("Always","0")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Frequently","1")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Sometimes","2")
train_csv['CAEC']= train_csv['CAEC'].str.replace("no","3")

test_csv['CAEC']= test_csv['CAEC'].str.replace("Always","0")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Frequently","1")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Sometimes","2")
test_csv['CAEC']= test_csv['CAEC'].str.replace("no","3")
#print(np.unique(train_csv['CAEC'], return_counts= True))
#print(np.unique(test_csv['CAEC'], return_counts= True))


#print(np.unique(test_csv['SMOKE'], return_counts= True))
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("yes","0")
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("no","1")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("yes","0")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("no","1")

#print(np.unique(train_csv['SMOKE'], return_counts= True))
#print(np.unique(test_csv['SMOKE'], return_counts= True))

#print(np.unique(train_csv['SCC'], return_counts= True))
train_csv['SCC']= train_csv['SCC'].str.replace("yes","0")
train_csv['SCC']= train_csv['SCC'].str.replace("no","1")
test_csv['SCC']= test_csv['SCC'].str.replace("yes","0")
test_csv['SCC']= test_csv['SCC'].str.replace("no","1")
#print(np.unique(test_csv['SCC'], return_counts= True))


#print(np.unique(test_csv['CALC'], return_counts= True))
test_csv['CALC']= test_csv['CALC'].str.replace("Always","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Frequently","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Sometimes","2")
test_csv['CALC']= test_csv['CALC'].str.replace("no","3")

#print(np.unique(train_csv['CALC'], return_counts= True))
train_csv['CALC']= train_csv['CALC'].str.replace("Always","0")
train_csv['CALC']= train_csv['CALC'].str.replace("Frequently","1")
train_csv['CALC']= train_csv['CALC'].str.replace("Sometimes","2")
train_csv['CALC']= train_csv['CALC'].str.replace("no","3")
#print(np.unique(train_csv['CALC'], return_counts= True))


#print(np.unique(train_csv['MTRANS'], return_counts= True))
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Automobile","0")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Bike","1")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Motorbike","2")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Public_Transportation","3")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Walking","4")

test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Automobile","0")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Bike","1")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Motorbike","2")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Public_Transportation","3")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Walking","4")



#print(np.unique(train_csv['MTRANS'], return_counts= True))
#print(np.unique(test_csv['MTRANS'], return_counts= True))


#print(test_csv.isnull().sum()) #없음.
#print(train_csv.isnull().sum()) #없음.

# (['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad', 'BMI'],


x = train_csv.drop(['NObeyesdad'], axis = 1)



y = train_csv['NObeyesdad']
y = le.fit_transform(y)
#test_csv = test_csv.drop(['FAVC'], axis = 1)

y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse= False)
y = ohe.fit_transform(y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
import time


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=42, stratify= y)
#5 9158 19 9145

# scaler = StandardScaler()
scaler = MinMaxScaler()
 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델구성
import warnings
warnings.filterwarnings('ignore')

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 18])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.random_normal([18,500]))
b1 = tf.compat.v1.Variable(tf.zeros([500]))
layer1 = tf.compat.v1.matmul(x, w1) + b1 

w2 = tf.compat.v1.Variable(tf.random_normal([500,400]))
b2 = tf.compat.v1.Variable(tf.zeros([400]))
layer2 = tf.compat.v1.matmul(layer1, w2) + b2 

w3 = tf.compat.v1.Variable(tf.random_normal([400,300]))
b3 = tf.compat.v1.Variable(tf.zeros([300]))
layer3 = tf.compat.v1.matmul(layer2, w3) + b3 

w4 = tf.compat.v1.Variable(tf.random_normal([300,200]))
b4 = tf.compat.v1.Variable(tf.zeros([200]))
layer4 = tf.compat.v1.matmul(layer3, w4) + b4 

w5 = tf.compat.v1.Variable(tf.random_normal([200,100]))
b5 = tf.compat.v1.Variable(tf.zeros([100]))
layer5 = tf.compat.v1.matmul(layer4, w5) + b5

w6 = tf.compat.v1.Variable(tf.random_normal([100,50]))
b6 = tf.compat.v1.Variable(tf.zeros([50]))
layer6 = tf.compat.v1.matmul(layer5, w6) + b6 
 
w7 = tf.compat.v1.Variable(tf.random_normal([50,7]))
b7 = tf.compat.v1.Variable(tf.zeros([7]))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer6, w7) + b7) 




#3-1. 컴파일

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis =1 ))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(loss)

from sklearn.metrics import r2_score, accuracy_score

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train],
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

# acc :  0.7480732177263969


