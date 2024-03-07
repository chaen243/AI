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
from sklearn.ensemble import BaggingRegressor
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

model = BaggingRegressor(LinearSVR(),
                         n_estimators= 10,
                         n_jobs= -1,
                        #  bootstrap= True,
                         bootstrap = False,
                         random_state=123
                         )


#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()



#4. 결과
print('score :', model.score(x_test, y_test))
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2_score :", r2)
print("걸린시간 :", round(end_time - start_time, 2), "초")

#true
# score : 0.5954545108979076

#false
# score : 0.5961812830848994