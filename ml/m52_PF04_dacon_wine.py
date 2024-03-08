#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier



#1. 데이터
path = "C:\\_data\\daicon\\wine\\"



train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
print(x)
y = train_csv['quality']-3





x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=67, shuffle=True, stratify= y) #67



scaler = StandardScaler()
#scaler = MinMaxScaler()
 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = pf.fit_transform(x_train)
x_test_poly = pf.transform(x_test)

#2. 모델
model = RandomForestClassifier()
model2 = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)
model2.fit(x_train_poly, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
y_pred2 = model2.predict(x_test_poly)

print('일반모델 acc :', accuracy_score(y_test, y_pred))
print('PF acc :', accuracy_score(y_test, y_pred2))

# 일반모델 acc : 0.6781818181818182
# PF acc : 0.6981818181818182