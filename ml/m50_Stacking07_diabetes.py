#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor


#1. 데이터

x,y = load_diabetes(return_X_y=True)



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle= True)#, stratify=y)



# scaler = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LinearRegression()

model = StackingRegressor(
    estimators=[('LR',lr),('RF',rf),('XGB',xgb)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs= -1,
    cv=5, 
)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, r2_score

y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('스태킹 r2 :', r2_score(y_test, y_pred))

#true
# score : 0.14743527046554583

#false
# score : -0.019377803909714553

#voting
# score : 0.3554196454074976


# model.score : 0.43471370554620836
# 스태킹 r2 : 0.43471370554620836

