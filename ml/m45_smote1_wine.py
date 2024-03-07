

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense



#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets['target']

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
#print(pd.value_counts(y))

print(y)
print("=======================================")
x = x[:-35]
y = y[:-35]
print(y)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 13], dtype=int64)) #증폭을 위해 임의로 자른것.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=123, stratify= y,)



#2. 모델

# model = Sequential()
# model.add(Dense(10, input_dim = 13))
# model.add(Dense(3, activation= 'softmax'))

# #3. 컴파일, 훈련
# model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
# #다중분류에서는 원핫을 하지 않아도 'sparse_categorical_crossentropy' 사용하면 됨.
# #원핫을 한 상태에서는 사용하면 에러남!

# model.fit(x_train, y_train, epochs = 100, batch_size=32, validation_split=0.2 )

# #4. 평가, 예측
# results = model.evaluate(x_test, y_test) 
# print('loss :', results[0])
# print('acc :', results[1])
# y_predict= model.predict(x_test)
# #print(y_test) 원핫x
# #print(y_predict) 원핫o
# y_predict = np.argmax(y_predict,axis=1)
# f1 = f1_score(y_test,y_predict,average='macro') # 다중분류는 average에서 선택한 방법으로 사용.
# acc = accuracy_score(y_test, y_predict)
# print('f1:', f1) #원래는 이진분류용으로 나오는 지표

# acc : 0.7777777910232544
# f1: 0.5661764705882353


############################### smote ###############################

print('==================smote============')
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런', sk.__version__)

smote = SMOTE(random_state=123)
x_train, y_train =smote.fit_resample(x_train, y_train)

print(pd.value_counts(y_train)) # 0    53 1    53 2    53

#2. 모델

model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(3, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
#다중분류에서는 원핫을 하지 않아도 'sparse_categorical_crossentropy' 사용하면 됨.
#원핫을 한 상태에서는 사용하면 에러남!

model.fit(x_train, y_train, epochs = 100, batch_size=32, validation_split=0.2 )

#4. 평가, 예측
results = model.evaluate(x_test, y_test) 
print('loss :', results[0])
print('acc :', results[1])
y_predict= model.predict(x_test)
#print(y_test) 원핫x
#print(y_predict) 원핫o
y_predict = np.argmax(y_predict,axis=1)
f1 = f1_score(y_test,y_predict,average='macro') # 다중분류는 average에서 선택한 방법으로 사용.
acc = accuracy_score(y_test, y_predict)
print('f1:', f1) #원래는 이진분류용으로 나오는 지표

#acc : 0.8055555820465088
#f1: 0.597090449082859