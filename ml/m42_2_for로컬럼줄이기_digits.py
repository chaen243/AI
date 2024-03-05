import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


datasets = load_digits() 
x = datasets.data
y = datasets.target

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
          verbose = 1,
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
    
    
# acc_score : 0.925
# 남은피처 갯수: 63, 0.925
# 남은피처 갯수: 62, 0.9444444444444444
# 남은피처 갯수: 61, 0.9305555555555556
# 남은피처 갯수: 60, 0.9388888888888889
# 남은피처 갯수: 59, 0.9361111111111111
# 남은피처 갯수: 58, 0.9416666666666667
# 남은피처 갯수: 57, 0.9305555555555556
# 남은피처 갯수: 56, 0.9388888888888889
# 남은피처 갯수: 55, 0.9222222222222223
# 남은피처 갯수: 54, 0.9333333333333333
# 남은피처 갯수: 53, 0.9388888888888889
# 남은피처 갯수: 52, 0.9444444444444444
# 남은피처 갯수: 51, 0.9388888888888889
# 남은피처 갯수: 50, 0.9388888888888889
# 남은피처 갯수: 49, 0.9305555555555556
# 남은피처 갯수: 48, 0.9416666666666667
# 남은피처 갯수: 47, 0.9305555555555556
# 남은피처 갯수: 46, 0.9361111111111111
# 남은피처 갯수: 45, 0.9305555555555556
# 남은피처 갯수: 44, 0.9277777777777778
# 남은피처 갯수: 43, 0.9333333333333333
# 남은피처 갯수: 42, 0.9305555555555556
# 남은피처 갯수: 41, 0.9277777777777778
# 남은피처 갯수: 40, 0.9305555555555556
# 남은피처 갯수: 39, 0.9416666666666667
# 남은피처 갯수: 38, 0.9361111111111111
# 남은피처 갯수: 37, 0.9222222222222223
# 남은피처 갯수: 36, 0.9166666666666666
# 남은피처 갯수: 35, 0.9305555555555556
# 남은피처 갯수: 34, 0.9388888888888889
# 남은피처 갯수: 33, 0.9305555555555556
# 남은피처 갯수: 32, 0.9305555555555556
# 남은피처 갯수: 31, 0.9222222222222223
# 남은피처 갯수: 30, 0.9222222222222223
# 남은피처 갯수: 29, 0.9166666666666666
# 남은피처 갯수: 28, 0.9333333333333333
# 남은피처 갯수: 27, 0.9305555555555556
# 남은피처 갯수: 26, 0.9222222222222223
# 남은피처 갯수: 25, 0.9277777777777778
# 남은피처 갯수: 24, 0.9138888888888889
# 남은피처 갯수: 23, 0.9305555555555556
# 남은피처 갯수: 22, 0.9333333333333333
# 남은피처 갯수: 21, 0.9111111111111111
# 남은피처 갯수: 20, 0.925
# 남은피처 갯수: 19, 0.8972222222222223
# 남은피처 갯수: 18, 0.9111111111111111
# 남은피처 갯수: 17, 0.9083333333333333
# 남은피처 갯수: 16, 0.9138888888888889
# 남은피처 갯수: 15, 0.8666666666666667
# 남은피처 갯수: 14, 0.9083333333333333
# 남은피처 갯수: 13, 0.8944444444444445
# 남은피처 갯수: 12, 0.875
# 남은피처 갯수: 11, 0.8694444444444445
# 남은피처 갯수: 10, 0.8722222222222222
# 남은피처 갯수: 9, 0.8638888888888889
# 남은피처 갯수: 8, 0.8388888888888889
# 남은피처 갯수: 7, 0.7583333333333333
# 남은피처 갯수: 6, 0.7277777777777777
# 남은피처 갯수: 5, 0.6555555555555556
# 남은피처 갯수: 4, 0.6305555555555555
# 남은피처 갯수: 3, 0.5027777777777778
# 남은피처 갯수: 2, 0.3416666666666667
# 남은피처 갯수: 1, 0.2722222222222222    
    
    
    
