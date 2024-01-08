import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터
path = "c:\\dacon\\ddarung\\"
#print(path + "aaa.csv") 문자 그대로 보여줌.

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission_csv = pd.read_csv(path + "submission.csv")

print(train_csv.shape)                  # (1459, 10)
print(test_csv.shape)                   # (715, 9)
print(submission_csv.shape)             # (715, 2)

print(train_csv.columns)                # id, 컬럼명은 데이터가 아니라 index일뿐
#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']

print(train_csv.describe())

##########결측치 처리 ###########33
print(train_csv.isnull().sum())
print(train_csv.isna().sum())
train_csv = train_csv.fillna(train_csv.mean())
train_csv = train_csv.dropna() #결측치가 하나라도 있을땐 행 전체 삭제.
test_csv = test_csv.fillna(test_csv.mean())

##########x와 y 분리하기 ########33
x = train_csv.drop(['count'], axis= 1)
y = train_csv['count']

print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, shuffle= False, random_state= 1004)
#print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
#print(y_train.shape, y_test.shape) #(929,) (399,)

#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim = 9))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(512))

#3. 컴파일, 훈련
model.compile  (loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit (x_train, y_train, epochs= 500, batch_size = 10)
end_time = time.time()

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)

print("로스 :", loss)
print("R2 스코어 :", r2)
print("걸린시간 :", round(end_time - start_time, 2), "초")
