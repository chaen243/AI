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
model = BaggingClassifier(LogisticRegression(),
                          n_estimators= 10,
                          n_jobs= -1,
                          random_state= 777,
                          bootstrap=True, #디폴트, 중복허용! (과적합 방지!)
                        #   bootstrap=False #중복 허용하지 않음xxxx
                          )

#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

##########################

# False
# 최종점수 0.7185184547731126
# acc_score : 0.7185184547731126

# True
# 최종점수 0.7200416512482466
# acc_score : 0.7200416512482466
