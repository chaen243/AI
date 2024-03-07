#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
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
 
model = BaggingRegressor(LinearSVC(),
                         n_jobs= -1,
                         n_estimators= 10,
                        #  bootstrap= True,
                         bootstrap= False,
                         random_state= 123
                         )    



#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측

from sklearn.metrics import accuracy_score

print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)


print("걸린시간 :", round(end_time - start_time, 2), "초")

#true
# score : 0.14743527046554583

#false
# score : -0.019377803909714553