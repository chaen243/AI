#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier



#1. 데이터
path = "C:\\_data\\daicon\\wine\\"



train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
print(x)
y = train_csv['quality']-3





x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=67, shuffle=True, stratify= y) #67



scaler = StandardScaler()
#scaler = MinMaxScaler()
 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델
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


model = XGBClassifier(tree_method = 'hist', device = "cuda")
model.set_params( **Params)








#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)
y_predict = model.predict(x_test)

#print(y_predict)
print(y_predict.shape, y_test.shape) #(1650, 2) (1650, 2)

y_submit = model.predict(test_csv)

import time

y_submit = (y_submit)+3
#print(y_test, y_predict)
result = accuracy_score(y_test, y_predict)
submission_csv['quality'] = y_submit
acc = accuracy_score(y_predict, y_test)
f1 = f1_score(y_predict, y_test, average= 'macro')

ltm = time.localtime(time.time())
print("acc :", acc)
print("f1 :", f1)

save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path+f"submission_{save_time}e.csv", index=False)




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
    
# acc : 0.6872727272727273
# f1 : 0.5049204255238603
# 남은피처 갯수: 11, 0.6836363636363636
# 남은피처 갯수: 10, 0.6854545454545454
# 남은피처 갯수: 9, 0.68
# 남은피처 갯수: 8, 0.6581818181818182
# 남은피처 갯수: 7, 0.6618181818181819
# 남은피처 갯수: 6, 0.6436363636363637
# 남은피처 갯수: 5, 0.6418181818181818
# 남은피처 갯수: 4, 0.6254545454545455
# 남은피처 갯수: 3, 0.6327272727272727
# 남은피처 갯수: 2, 0.509090909090909
# 남은피처 갯수: 1, 0.44181818181818183