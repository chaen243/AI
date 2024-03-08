#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
import time


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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= False, random_state= 6) #399 #1048 #6


scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)





#2. 모델 구성

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

pf = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = pf.fit_transform(x_train)
x_test_poly = pf.transform(x_test)

#2. 모델
model = RandomForestRegressor()
model2 = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)
model2.fit(x_train_poly, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
y_pred2 = model2.predict(x_test_poly)

print('일반모델 r2 :', r2_score(y_test, y_pred))
print('PF r2 :', r2_score(y_test, y_pred2))

# 일반모델 r2 : 0.7599832100915234
# PF r2 : 0.764123931434631

