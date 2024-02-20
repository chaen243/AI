import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
import tensorflow as tf


#1. 데이터
path = "c:\\_data\\daicon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)

print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

print(x, y)
print(x.shape, y.shape) #(5497, 12) (5497,)
print(np.unique(y, return_counts= True)) # (array([3, 4, 5, 6, 7, 8, 9], array([  26,  186, 1788, 2416,  924,  152,    5]

y = pd.get_dummies(y)
print(y)


x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle= True, random_state=364, stratify= y)


#2. 모델구성

model = Sequential()
model.add(Dense(100, input_dim = 12))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(7, activation = 'softmax'))

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)




#3. 컴파일, 훈련
model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor= 'val_acc', mode= 'max', patience= 200, verbose= 0, restore_best_weights= True)
start_time = time.time()
hist = model.fit (x_train, y_train, epochs = 300, batch_size= 10, validation_split= 0.3, callbacks= [es] )
end_time = time.time()

#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("acc :", results[1])

y_predict = model.predict(x_test)

y_submit = model.predict(test_csv)
print(y_predict)
print(y_predict.shape, y_test.shape)
print(y_test, y_predict)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
y_submit = np.argmax(y_submit, axis=1)+3

result = accuracy_score(y_test, y_predict)
submission_csv['quality'] = y_submit
print(y_submit)

ltm = time.localtime(time.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}.csv", index=False)
acc = accuracy_score(y_predict, y_test)
print("acc :", acc)

print("걸린 시간 :", round(end_time - start_time, 2), "초" )
