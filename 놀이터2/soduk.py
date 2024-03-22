import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, Normalizer
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, StratifiedKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from keras.callbacks import ReduceLROnPlateau

import time
import random


random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)


#1. 데이터

path = 'D:\money\\'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
#print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
#print(test_csv)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
#print(submission_csv)

# print(train_csv.shape) #(20000, 22)
# print(test_csv.shape) #(10000, 21)
# print(submission_csv.shape) #(10000, 2)

#print(train_csv.columns)
# ['Age', 'Gender', 'Education_Status', 'Employment_Status',
#        'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
#        'Hispanic_Origin', 'Martial_Status', 'Household_Status',
#        'Household_Summary', 'Citizenship', 'Birth_Country',
#        'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
#        'Gains', 'Losses', 'Dividends', 'Income_Status', 'Income'],

le = LabelEncoder()
le.fit(train_csv['Gender'])
train_csv['Gender'] = le.transform(train_csv['Gender'])
le.fit(test_csv['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])
#print(train_csv)

le.fit(train_csv['Education_Status'])
train_csv['Education_Status'] = le.transform(train_csv['Education_Status'])
le.fit(test_csv['Education_Status'])
test_csv['Education_Status'] = le.transform(test_csv['Education_Status'])
#print(train_csv)

le.fit(train_csv['Employment_Status'])
train_csv['Employment_Status'] = le.transform(train_csv['Employment_Status'])
le.fit(test_csv['Employment_Status'])
test_csv['Employment_Status'] = le.transform(test_csv['Employment_Status'])


le.fit(train_csv['Industry_Status'])
train_csv['Industry_Status'] = le.transform(train_csv['Industry_Status'])
le.fit(test_csv['Industry_Status'])
test_csv['Industry_Status'] = le.transform(test_csv['Industry_Status'])


le.fit(train_csv['Occupation_Status'])
train_csv['Occupation_Status'] = le.transform(train_csv['Occupation_Status'])
le.fit(test_csv['Occupation_Status'])
test_csv['Occupation_Status'] = le.transform(test_csv['Occupation_Status'])

le.fit(train_csv['Race'])
train_csv['Race'] = le.transform(train_csv['Race'])
le.fit(test_csv['Race'])
test_csv['Race'] = le.transform(test_csv['Race'])

le.fit(train_csv['Hispanic_Origin'])
train_csv['Hispanic_Origin'] = le.transform(train_csv['Hispanic_Origin'])
le.fit(test_csv['Hispanic_Origin'])
test_csv['Hispanic_Origin'] = le.transform(test_csv['Hispanic_Origin'])

le.fit(train_csv['Martial_Status'])
train_csv['Martial_Status'] = le.transform(train_csv['Martial_Status'])
le.fit(test_csv['Martial_Status'])
test_csv['Martial_Status'] = le.transform(test_csv['Martial_Status'])

le.fit(train_csv['Household_Status'])
train_csv['Household_Status'] = le.transform(train_csv['Household_Status'])
le.fit(test_csv['Household_Status'])
test_csv['Household_Status'] = le.transform(test_csv['Household_Status'])

#print(train_csv['Household_Status'])


##########Citizen 값 병합#################
train_csv['Citizenship'] = train_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
test_csv['Citizenship'] = test_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
# print(np.unique(train_csv['Citizenship'], return_counts= True))

le.fit(train_csv['Citizenship'])
train_csv['Citizenship'] = le.transform(train_csv['Citizenship'])
le.fit(test_csv['Citizenship'])
test_csv['Citizenship'] = le.transform(test_csv['Citizenship'])
# print(np.unique(train_csv['Citizenship'], return_counts= True))
# print(np.unique(test_csv['Citizenship'], return_counts= True))


###########Birth_Country 병합############
#print(np.unique(train_csv['Birth_Country'], return_counts= True))

to_replace_values1 = ['Greece''Hungary','Poland', 'Scotland','Ireland', 'Italy', 'Canada','England','France', 'Germany']
target_value1 = 'Italy'

to_replace_values2 = ['Trinadad&Tobago','Haiti','Holand-Netherlands','Nicaragua','Dominican-Republic','Yugoslavia','Honduras','Portugal', 
                      'Puerto-Rico', 'Jamaica', 'Laos', 'Panama', 'Peru','Mexico','Ecuador', 'El-Salvador','Guatemala','Columbia', 'Cuba','Cambodia']
target_value2 = 'Mexico'

to_replace_values3 = ['Thailand','Philippines','Vietnam','India','Iran']
target_value3 = 'Thailand'

to_replace_values4 = ['South Korea', 'China','Taiwan','Hong Kong','Japan']
target_value4 = 'Soutn Korea'

to_replace_with_nan = ['Outlying-U S (Guam USVI etc)','Unknown']


# .replace()를 사용하여 여러 값을 한 가지 값으로 변경
train_csv['Birth_Country'] = train_csv['Birth_Country'].replace(to_replace_values1, target_value1)
test_csv['Birth_Country'] = test_csv['Birth_Country'].replace(to_replace_values1, target_value1)

train_csv['Birth_Country'] = train_csv['Birth_Country'].replace(to_replace_values2, target_value2)
test_csv['Birth_Country'] = test_csv['Birth_Country'].replace(to_replace_values2, target_value2)

train_csv['Birth_Country'] = train_csv['Birth_Country'].replace(to_replace_values3, target_value3)
test_csv['Birth_Country'] = test_csv['Birth_Country'].replace(to_replace_values3, target_value3)

train_csv['Birth_Country'] = train_csv['Birth_Country'].replace(to_replace_values4, target_value4)
test_csv['Birth_Country'] = test_csv['Birth_Country'].replace(to_replace_values4, target_value4)

train_csv['Birth_Country'] = train_csv['Birth_Country'].replace(to_replace_with_nan, np.nan)
test_csv['Birth_Country'] = test_csv['Birth_Country'].replace(to_replace_with_nan, np.nan)


le.fit(train_csv['Birth_Country'])
train_csv['Birth_Country'] = le.transform(train_csv['Birth_Country'])
le.fit(test_csv['Birth_Country'])
test_csv['Birth_Country'] = le.transform(test_csv['Birth_Country'])

print(np.unique(test_csv['Birth_Country'], return_counts= True))

le.fit(train_csv['Tax_Status'])
train_csv['Tax_Status'] = le.transform(train_csv['Tax_Status'])
le.fit(test_csv['Tax_Status'])
test_csv['Tax_Status'] = le.transform(test_csv['Tax_Status'])


le.fit(train_csv['Income_Status'])
train_csv['Income_Status'] = le.transform(train_csv['Income_Status'])
le.fit(test_csv['Income_Status'])
test_csv['Income_Status'] = le.transform(test_csv['Income_Status'])

le.fit(train_csv['Birth_Country (Father)'])
train_csv['Birth_Country (Father)'] = le.transform(train_csv['Birth_Country (Father)'])
le.fit(test_csv['Birth_Country (Father)'])
test_csv['Birth_Country (Father)'] = le.transform(test_csv['Birth_Country (Father)'])

le.fit(train_csv['Birth_Country (Mother)'])
train_csv['Birth_Country (Mother)'] = le.transform(train_csv['Birth_Country (Mother)'])
le.fit(test_csv['Birth_Country (Mother)'])
test_csv['Birth_Country (Mother)'] = le.transform(test_csv['Birth_Country (Mother)'])


# print(test_csv.isnull().sum()) #없음.
# print(train_csv.isnull().sum()) #없음.


# 삭제할 컬럼
# Household_Summary
# Income_Status

x = train_csv.drop(['Household_Summary','Income'], axis=1)
test_csv = test_csv.drop(['Household_Summary'], axis=1)

# print(x)

y = train_csv['Income']

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.85,  shuffle= True, random_state= 1117)

scaler = MinMaxScaler()
#scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit_transform(x_train)
scaler.transform(x_test)
test_csv= scaler.transform(test_csv)

parameters = {
    'n_estimators' : 5000,
    'learning_rate' : 0.1,
    'max_depth' : 35,
    'num_leaves' : 30,
    'min_child_weight' : 1,
    'min_child_samples' : 10,
    'subsample' : 0.7,
    'min_sample_leaf' : 3
    
}

model = XGBRegressor(bootstrap = True)
model.set_params(early_stopping_round= 500, #반환 디폴트 트루!
**parameters)

# parameters = [
#     {"n_estimators": [18000, 20000], "max_depth": [30, 10, 20], "min_samples_leaf": [3, 10], "subsample": [0.7, 0.8, 0,9]},
#     {"max_depth": [30, 20, 10, 12], "min_samples_leaf": [3, 5, 7, 10], 'subsample': [0.7, 0.8, 0,9]},
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10], 'subsample': [0.7, 0.8, 0,9]},
# ]    

# n_splits=5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# model = HalvingGridSearchCV(LGBMRegressor(), 
#                      parameters, 
#                      cv=kfold, 
#                      verbose=1, 
#                      random_state=42,
#                      refit= True, #디폴트 트루~
#                      n_jobs=-1) #CPU 다 쓴다!

# model.set_params(early_stopping_rounds= 100, #반환 디폴트 트루!
#     **parameters)


#3. 훈련

start_time = time.time()
model.fit(x_train, y_train, 
          eval_set=[(x_train, y_train),(x_test, y_test)],
          #verbose= 1, #2부터는 step으로 보여줌
          eval_metric = 'rmse' 
          )

end_time = time.time()



#4. 평가, 예측
from sklearn.metrics import r2_score
  
results = model.score(x_test, y_test)
print('최종점수 :', results) #회귀 디폴트 rmse

y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))


print("걸린시간 :", round(end_time - start_time, 2), "초")


y_submit = model.predict(test_csv)

import datetime
dt = datetime.datetime.now()
submission_csv['Income'] = y_submit

submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_.csv",index=False)



