import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
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

Params = {'n_estimators' : 1000,
          'learning_rate' : 0.01,
          'max_depth' : 3,
          'gamma' : 0,
          'min_child_weight' : 0,
          'subsample' : 0.4,
          'colsample_bytree' : 0.8,
          'colsample_lylevel' : 0.7,
          'colsample_bynode' : 1,
          'reg_alpha' : 0,
          'reg_lamba' : 1,
          'random_state' : 3377,
          'verbose' : 0        
          }

#2. 모델


model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **Params)
#3. 훈련
model.fit(x_train, y_train, 
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 0,
          eval_metric = 'merror')

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

##########################

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
          eval_metric = 'merror')
      
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(f"남은피처 갯수: {x_train.shape[1]}, {acc}")
    else:
        print("No features left.")
        break
    
    
# acc_score : 0.6890441726977788
# 남은피처 갯수: 53, 0.7477603848437648
# 남은피처 갯수: 52, 0.692081960018244
# 남은피처 갯수: 51, 0.6928908892197275
# 남은피처 갯수: 50, 0.6963847749197525
# 남은피처 갯수: 49, 0.6922884951335163
# 남은피처 갯수: 48, 0.6928908892197275
# 남은피처 갯수: 47, 0.6954467612712236
# 남은피처 갯수: 46, 0.6940784661325439
# 남은피처 갯수: 45, 0.6974518730153266
# 남은피처 갯수: 44, 0.7012813782776692
# 남은피처 갯수: 43, 0.6957049301653141
# 남은피처 갯수: 42, 0.6952918599347693
# 남은피처 갯수: 41, 0.6973399998278874
# 남은피처 갯수: 40, 0.692649931585243
# 남은피처 갯수: 39, 0.6938977479066806
# 남은피처 갯수: 38, 0.6893883978898996
# 남은피처 갯수: 37, 0.7471924132767657
# 남은피처 갯수: 36, 0.6959286765401926
# 남은피처 갯수: 35, 0.6942677899882103
# 남은피처 갯수: 34, 0.6972453379000542
# 남은피처 갯수: 33, 0.6963159298813284
# 남은피처 갯수: 32, 0.6929941567773638
# 남은피처 갯수: 31, 0.6860408078965259
# 남은피처 갯수: 30, 0.6953520993433905
# 남은피처 갯수: 29, 0.690145693312565
# 남은피처 갯수: 28, 0.691582833489669
# 남은피처 갯수: 27, 0.6898358906396564
# 남은피처 갯수: 26, 0.6923143120229254
# 남은피처 갯수: 25, 0.6933039594502723
# 남은피처 갯수: 24, 0.6919184530519866
# 남은피처 갯수: 23, 0.6992160271249451
# 남은피처 갯수: 22, 0.6974949011643418
# 남은피처 갯수: 21, 0.696161028544874
# 남은피처 갯수: 20, 0.6847327521664673
# 남은피처 갯수: 19, 0.6920905656480469
# 남은피처 갯수: 18, 0.6931490581138181
# 남은피처 갯수: 17, 0.69264132595544
# 남은피처 갯수: 16, 0.6994311678700206
# 남은피처 갯수: 15, 0.6972625491596602
# 남은피처 갯수: 14, 0.6898100737502474
# 남은피처 갯수: 13, 0.6916947066771082
# 남은피처 갯수: 12, 0.6914709603022298
# 남은피처 갯수: 11, 0.6883385110539315
# 남은피처 갯수: 10, 0.6772802767570545
# 남은피처 갯수: 9, 0.6847671746856794
# 남은피처 갯수: 8, 0.6830374430952729
# 남은피처 갯수: 7, 0.682607161605122
# 남은피처 갯수: 6, 0.6821768801149712
# 남은피처 갯수: 5, 0.6777535863962204
# 남은피처 갯수: 4, 0.675524728277239
# 남은피처 갯수: 3, 0.6745608977393011
# 남은피처 갯수: 2, 0.6694749705257179
# 남은피처 갯수: 1, 0.5197025894340077