import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import LinearSVR



#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델구성
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor

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
from sklearn.metrics import r2_score

y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('스태킹 r2 :', r2_score(y_test, y_pred))


#true
# score : 0.5954545108979076

#false
# score : 0.5961812830848994

#voting
# score : 0.6676037351317317

# model.score : 0.646738778347236
# 스태킹 r2 : 0.646738778347236