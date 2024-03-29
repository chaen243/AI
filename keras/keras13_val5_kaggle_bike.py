import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(submission_csv.shape) # (6493, 2)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.73, shuffle = False, random_state=662)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620, ) (3266, )

#2. 모델구성
model = Sequential()
model.add(Dense(512, input_dim = 8, activation='relu')) #활성화함수!
model.add(Dense(256,))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1 ))

#3. 컴파일, 훈련
model.compile (loss = 'mse', optimizer= 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs= 400, batch_size = 32, validation_split= 0.32, verbose= 2)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape) #(6493, 1)
#print(submission_csv.shape) #(6493, 2)


print("MSE :", loss)
print("R2 스코어 :", r2)
print("걸린 시간:", round(end_time - start_time, 2), "초")

submission_csv['count'] = y_submit

print(submission_csv)

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}.csv", index=False)

#submission_csv.to_csv(path + "submission_29.csv", index= False)
print("음수갯수 :", submission_csv[submission_csv['count']<0].count())


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict)
print("RMSLE :", rmsle)

#MSE : 23175.111328125
#R2 스코어 : 0.27044473122031987
#RMSE :  152.23374956711748
#RMSLE : 1.3152084898668681