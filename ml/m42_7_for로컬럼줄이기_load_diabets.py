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
from sklearn.ensemble import RandomForestClassifier
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
    

#xgb파라미터
# 'n_estimator : 디폴트 100/ 1~inf/ 정수
'''learning_rate : 디폴트 0.3/ 0~1 /eta''' #적을수록 좋다
#max_depth 디폴트 6 / 0~inf / 정수 #트리의 깊이
# 'gamma' : 디폴트 0/ 0~inf #값이 클수록 데이터 손실이 생김 작을수록 많이 분할되지만 트리가 복잡해짐.
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100]/ 디폴트1/ 0~inf
# 'subsample' : [0, 0.1, 0.3~]/ 디폴트1/ 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3~~]/ 디폴트1/ 0~1
# 'colsample_bylevel' : [0, 0.1, 0.5]/ 디폴트1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.5]/ 디폴트1 / 0~1
# 'reg_alpha': [0, 0.1, 0.001]/ 디폴트 0 / 0 ~ inf/ L1 절대값 가중치 규제/ alpha #음수쪽 규제
# 'reg_lambda': [0, 0.1, 0.001]/ 디폴트 1 / 0 ~ inf/ L2 제곱 가중치 규제/ lambda 

Params = {'n_estimators' : 1500,
          'learning_rate' : 0.05,
          'max_depth' : 10,
          'gamma' : 0,
          'min_child_weight' : 0,
          'subsample' : 0.4,
          'colsample_bytree' : 0.8,
          'colsample_bynode' : 1,
          'reg_alpha' : 0,
          'random_state' : 1234,
          }


model = XGBRegressor(tree_method = 'hist', device = "cuda")
model.set_params( **Params)



#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측

from sklearn.metrics import accuracy_score

print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)


print("걸린시간 :", round(end_time - start_time, 2), "초")


for _ in range(len(x_train[0])):
    # 피처 중요도 계산
    feature_importances = model.feature_importances_
    
    # 가장 중요도가 낮은 피처의 인덱스 제거
    least_important_feature_index = np.argmin(feature_importances)
    x_train = np.delete(x_train, least_important_feature_index, axis=1)
    x_test = np.delete(x_test, least_important_feature_index, axis=1)
    
    # 모델 재학습 및 성능 평가
    if x_train.shape[1] > 0:
        model.fit(x_train, y_train, 
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 0,
          eval_metric = 'rmse')
      
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(f"남은피처 갯수: {x_train.shape[1]}, {acc}")
    else:
        print("No features left.")
        break
    

# score : 0.9089595375722543
# 남은피처 갯수: 17, 0.9099229287090559
# 남은피처 갯수: 16, 0.9104046242774566
# 남은피처 갯수: 15, 0.9070327552986512
# 남은피처 갯수: 14, 0.9046242774566474
# 남은피처 갯수: 13, 0.9026974951830443
# 남은피처 갯수: 12, 0.903179190751445
# 남은피처 갯수: 11, 0.8973988439306358
# 남은피처 갯수: 10, 0.8973988439306358
# 남은피처 갯수: 9, 0.8983622350674374
# 남은피처 갯수: 8, 0.8964354527938343
# 남은피처 갯수: 7, 0.8978805394990366
# 남은피처 갯수: 6, 0.896917148362235
# 남은피처 갯수: 5, 0.8771676300578035
# 남은피처 갯수: 4, 0.8684971098265896
# 남은피처 갯수: 3, 0.8636801541425819
# 남은피처 갯수: 2, 0.8516377649325626
# 남은피처 갯수: 1, 0.3511560693641618