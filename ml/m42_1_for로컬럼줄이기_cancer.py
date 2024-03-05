import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

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
          verbose = 1,
          eval_metric = 'logloss')

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
          eval_metric = 'logloss')
      
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(f"남은피처 갯수: {x_train.shape[1]}, {acc}")
    else:
        print("No features left.")
        break
    
    
    
    
#     acc_score : 0.9385964912280702
# 남은피처 갯수: 29, 0.9298245614035088
# 남은피처 갯수: 28, 0.9385964912280702
# 남은피처 갯수: 27, 0.9298245614035088
# 남은피처 갯수: 26, 0.9385964912280702
# 남은피처 갯수: 25, 0.9385964912280702
# 남은피처 갯수: 24, 0.9385964912280702
# 남은피처 갯수: 23, 0.9298245614035088
# 남은피처 갯수: 22, 0.9298245614035088
# 남은피처 갯수: 21, 0.9385964912280702
# 남은피처 갯수: 20, 0.9385964912280702
# 남은피처 갯수: 19, 0.9385964912280702
# 남은피처 갯수: 18, 0.9385964912280702
# 남은피처 갯수: 17, 0.9385964912280702
# 남은피처 갯수: 16, 0.9385964912280702
# 남은피처 갯수: 15, 0.9385964912280702
# 남은피처 갯수: 14, 0.9385964912280702
# 남은피처 갯수: 13, 0.9298245614035088
# 남은피처 갯수: 12, 0.9385964912280702
# 남은피처 갯수: 11, 0.9385964912280702
# 남은피처 갯수: 10, 0.9210526315789473
# 남은피처 갯수: 9, 0.9385964912280702
# 남은피처 갯수: 8, 0.9298245614035088
# 남은피처 갯수: 7, 0.9385964912280702
# 남은피처 갯수: 6, 0.9210526315789473
# 남은피처 갯수: 5, 0.9298245614035088
# 남은피처 갯수: 4, 0.9210526315789473
# 남은피처 갯수: 3, 0.9210526315789473
# 남은피처 갯수: 2, 0.9035087719298246
# 남은피처 갯수: 1, 0.8771929824561403