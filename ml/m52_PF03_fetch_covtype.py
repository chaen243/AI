import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
import time

warnings.filterwarnings('ignore')

#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = datasets.target-1

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify= y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

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

# 일반모델 acc : 0.9551388518368716
# PF acc : 0.9590974415462595


