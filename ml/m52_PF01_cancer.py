import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)


#2. 모델구성

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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


# 일반모델 acc : 0.9562841530054644
# PF acc : 0.9781420765027322
