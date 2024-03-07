import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
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
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = StackingClassifier(
    estimators=[('LR',lr),('RF',rf),('XGB',xgb)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs= -1,
    cv=5, 
)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('스태킹 acc :', accuracy_score(y_test, y_pred))


##########################

# False
# 최종점수 0.7185184547731126
# acc_score : 0.7185184547731126

# True
# 최종점수 0.7200416512482466
# acc_score : 0.7200416512482466

#voting soft
# 최종점수 0.9024896087020129

#voting hard
# 최종점수 0.8841510115917833

#stacking
# model.score : 0.9624880596886484
# 스태킹 acc : 0.9624880596886484
