#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.utils import custom_object_scope
import time
def rmse(y, y_predict):    
    return mean_squared_error(y, y_predict)**0.5


#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"
# print(path + "aaa.csv") 문자그대로 보여줌 c:\_data\daicon\ddarung\aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")
#print(submission_csv)

#print(train_csv.shape)      # (1459, 10)
#print(test_csv.shape)       # (715, 9)
#print(submission_csv.shape) # (715, 2)

#print(train_csv.columns)    # id,컬럼명(header)는 데이터x, index일뿐
# [      'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

#print(train_csv.info())
#print(test_csv.info())

#print(train_csv.describe()) #함수는 뒤에 괄호를 꼭 넣어야 실행이 됨. 데이터의정보가 나옴.

############결측치 처리 1. 제거 ##########
#print(train_csv.isnull().sum())
#print(train_csv.isna().sum()) (둘다 똑같음)
train_csv = train_csv.fillna(train_csv.min())  #결측치가 하나라도 있으면 행전체 삭제됨.
test_csv = test_csv.fillna(test_csv.mean())   # (0,mean)
#print(train_csv.isnull().sum())
#print(train_csv.info())
#print(train_csv.shape)      #(1328, 10)
#print(test_csv.info()) # 717 non-null


################# x와 y를 분리 ###########
x = train_csv.drop(['count'], axis=1)
#print(x)
y = train_csv['count']
#print(y)

print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle= False, random_state= 1002) #399 #1048 #6
#print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
#print(y_train.shape, y_test.shape) #(929,) (399,)

# 로스 : 2656.447021484375
#R2 스코어 : 0.6342668951889647
#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 9))
model.add(Dense(12))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))



    

#3. 컴파일, 훈련
model.compile (loss = 'mse' , optimizer = 'adam') 
start_time = time.time()
model.fit(x_train, y_train, epochs=543, batch_size= 10 )
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)

#print(submission_csv.shape)
print("로스 :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")





####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
submission_csv['count'] = y_submit
print(submission_csv)

submission_csv.to_csv(path + "submission__6.csv", index= False)
