#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, RandomForestRegressor, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
import time


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

# 일반모델 r2 : 0.3919916343683302
# PF r2 : 0.333251906266934

