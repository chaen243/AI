#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import time


#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"
# print(path + "aaa.csv") 문자그대로 보여줌 c:\_data\daicon\ddarung\aaa.csv
# pandas에서 1차원- Series, 2차원이상은 DataFrame이라고 함.

train_csv = pd.read_csv(path + "train.csv", index_col=['id']) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv") 
#print(submission_csv)


###########################결측치처리########################


train_csv = train_csv.fillna(train_csv.ffill())  #뒤 데이터로 채움
print(train_csv.isnull().sum())
test_csv = test_csv.fillna(test_csv.ffill())  #앞 데이터로 채움
print(test_csv.isnull().sum())

print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)      #(1328, 10)
print(test_csv.info()) # 717 non-null

###########################결측치처리########################




################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
#print(x)
y = train_csv['count']
#print(y)

###########################이상치처리########################


def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
    data = data.fillna(data.median())
    return data

###########################이상치처리########################




x = fit_outlier(x)



print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= False, random_state= 6) #399 #1048 #6


scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)





#2. 모델 구성
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


model = XGBRegressor()
model.set_params(**Params, eval_metric = 'rmse')

#3. 훈련
model.fit(x_train, y_train, eval_set = [(x_test, y_test)],  verbose = 0)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = r2_score(y_test, x_predict)
print("acc_score :", acc)


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
        r2 = r2_score(y_test, y_predict)
        print(f"남은피처 갯수: {x_train.shape[1]}, {r2}")
    else:
        print("No features left.")
        break
    


