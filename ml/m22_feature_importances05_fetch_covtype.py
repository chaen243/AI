import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier


import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import time
import tensorflow as tf
#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = datasets.target
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7,  shuffle= True, random_state= 398, stratify= y) #y의 라벨값을 비율대로 잡아줌 #회귀모델에서는 ㄴㄴ 분류에서만 가능
#print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
#print(y_train.shape, y_test.shape) #(7620, ) (3266, )
#print(np.unique(y_test, return_counts = True ))


# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler

# #mms = MinMaxScaler()
# mms = StandardScaler()
# #mms = MaxAbsScaler()
# #mms = RobustScaler()

# mms.fit(x_train)
# x_train= mms.transform(x_train)
# x_test= mms.transform(x_test)


#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        

#print("mms = StandardScaler")
#로스 : 0.2562239468097687
#정확도 : 0.8990269899368286

#print('#mms = MaxAbsScaler')
#로스 : 0.2306247502565384
#정확도 : 0.910214364528656

#print('#mms = RobustScaler')
#로스 : 0.23458202183246613
#정확도 : 0.9094054102897644

#minmax
# 로스 : 0.34160080552101135
# 정확도 : 0.8604621887207031

#model.score : 0.7131677987883238

#Linear
# model.score : 0.5243826877180099
# acc : 0.5243826877180099
# 걸린시간 : 348.6212000846863 초