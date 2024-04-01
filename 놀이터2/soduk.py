import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, Normalizer
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from keras.callbacks import ReduceLROnPlateau
#import optuna
from catboost import CatBoostRegressor
import pickle
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

#print(test_csv.isnull().sum()) #1개
#print(train_csv.isnull().sum()) #없음.

test_csv = test_csv.fillna(method= 'bfill')



##########Citizen 값 병합#################
train_csv['Citizenship'] = train_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
test_csv['Citizenship'] = test_csv['Citizenship'].apply(lambda x: 'Native' if 'Native' in x else x)
# print(np.unique(train_csv['Citizenship'], return_counts= True))

###########Birth_Country 병합############
#print(np.unique(train_csv['Birth_Country'], return_counts= True))

train_csv.loc[train_csv['Birth_Country'] != 'US', 'Birth_Country'] = 'not US'
train_csv.loc[train_csv['Birth_Country (Father)'] != 'US', 'Birth_Country (Father)'] = 'not US'
train_csv.loc[train_csv['Birth_Country (Mother)'] != 'US', 'Birth_Country (Mother)'] = 'not US'

test_csv.loc[test_csv['Birth_Country'] != 'US', 'Birth_Country'] = 'not US'
test_csv.loc[test_csv['Birth_Country (Father)'] != 'US', 'Birth_Country (Father)'] = 'not US'
test_csv.loc[test_csv['Birth_Country (Mother)'] != 'US', 'Birth_Country (Mother)'] = 'not US'



###########label encoder############
label_encoder_dict = {}
for label in train_csv:
    data = train_csv[label].copy()
    if data.dtypes == 'object':
        label_encoder = LabelEncoder()
        train_csv[label] = label_encoder.fit_transform(data)
        label_encoder_dict[label] = label_encoder
  
# # inverse_transform 작동 확인 완료
# for label, label_encoder in label_encoder_dict.items():
#     data = train_csv[label].copy()
#     train_csv[label] = label_encoder.inverse_transform(data)
# print(train_csv.head(10))
#

for label, encoder in label_encoder_dict.items():
    data = test_csv[label]
    test_csv[label] = encoder.transform(data)
#print(test_csv.head(10))

# print(test_csv.isna().sum())




# 삭제할 컬럼
# Household_Summary
# Income_Status




x = train_csv.drop(['Household_Summary','Income'], axis=1)
#x = train_csv.drop(['Income'], axis=1)

test_csv = test_csv.drop(['Household_Summary'], axis=1)


y = train_csv['Income']


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.85,  shuffle= True, random_state= 42)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit_transform(x_train)
scaler.transform(x_test)
test_csv= scaler.transform(test_csv)

# def objectiveXGB(trial):
#     param = {
#         'n_estimators' : trial.suggest_int('n_estimators', 500, 2000),
#         'max_depth' : trial.suggest_int('max_depth', 5, 15),
#         'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
#         'gamma' : trial.suggest_float('gamma', 1, 5.0),
#         'learning_rate' : trial.suggest_float('learning_rate', 0.0001, 0.008),
#         'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1),
#         'nthread' : -2,
#         'tree_method' : 'hist',
#         'device' : 'cuda',
#         'lambda' : trial.suggest_float('lambda', 1.0, 5.0),
#         'alpha' : trial.suggest_float('alpha', 1.0, 5.0),
#         'subsample' : trial.suggest_float('subsample', 0.5, 1),
#         #'random_state' : trial.suggest_int('random_state', 1, 10000)
#     }
    
#     # BEST PARAM:  {'n_estimators': 1557, 'max_depth': 9, 'min_child_weight': 8, 
#     #               'gamma': 3.903915892360949, 'learning_rate': 0.005244213833337055, 'colsample_bytree': 0.6497242994796232, 'lambda': 4.257154545398103, 'alpha': 1.2487172832917386, 'subsample': 1.0}
    
#     # BEST PARAM:  {'n_estimators': 1098, 'max_depth': 12, 'min_child_weight': 7,
#     #               'gamma': 2.6792433250176106, 'learning_rate': 0.003462313983240453, 'colsample_bytree': 0.9887926225119256, 'lambda': 1.7754130865891198, 'alpha': 2.308523947262986, 'subsample': 0.7882610347122883}
    
#     # BEST PARAM:  {'n_estimators': 1338, 'max_depth': 11, 'min_child_weight': 9,
#     #               'gamma': 1.2362621416480555, 'learning_rate': 0.0030242459366422907, 'colsample_bytree': 0.8917657962273478, 'lambda': 3.1461166883225427, 'alpha': 3.128671605110223, 'subsample': 0.6856006013705762}
#     # 학습 모델 생성
#     model = XGBRegressor(**param)
#     model = model.fit(x_train, y_train) # 학습 진행
    
#     # 모델 성능 확인
#     y_predict = model.predict(x_test)
#     score = r2_score(y_test, y_predict)
    
#     print('score : ', score)
#     return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveXGB, n_trials=30)

# best_params = study.best_params
# print("BEST PARAM: ",best_params)


# def objectiveLGBM(trial):
#     param = {
#         'n_estimators' : trial.suggest_int('n_estimators', 600, 1500),
#         'max_depth' : trial.suggest_int('max_depth', 10, 50),
#         'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
#         'gamma' : trial.suggest_float('gamma', 0, 3.0),
#         'learning_rate' : trial.suggest_float('learning_rate', 0.0001, 0.05),
#         'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1),
#         'num_leaves' : trial.suggest_int('num_leaves', 2, 50),
#         'nthread' : -2,
#         # 'tree_method' : 'hist',
#         # 'device' : 'cuda',
#         'lambda' : trial.suggest_float('lambda', 0, 3.0),
#         'alpha' : trial.suggest_float('alpha', 0, 5.0),
#         'subsample' : trial.suggest_float('subsample', 0.5, 0.9),
#         #'random_state' : trial.suggest_int('random_state', 1, 10000)
#     }
#     # 학습 모델 생성
#     model = LGBMRegressor(**param, random_state = 42)
#     model = model.fit(x_train, y_train, ) # 학습 진행
    
#     # 모델 성능 확인
#     y_predict = model.predict(x_test)
#     score = r2_score(y_test, y_predict)
    
#     print('score : ', score)
#     return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveLGBM, n_trials=300)

# best_params = study.best_params
# print("BEST PARAM: ",best_params) 

# def objectiveCATBOOST(trial):
#     param = {
#         'n_estimators' : trial.suggest_int('n_estimators', 600, 1500),
#         'max_depth' : trial.suggest_int('max_depth', 10, 16),
#         'learning_rate' : trial.suggest_float('learning_rate', 0.0001, 0.05),
#         'bootstrap_type' : trial.suggest_categorical('bootstrap_type', ['Bayesian','Bernoulli','MVS']),
#         'boosting_type' : trial.suggest_categorical('boosting_type', ['Ordered']),
#         'task_type' : 'GPU',
#         #'subsample' : trial.suggest_float('subsample', 0.5, 0.9),
#         #'random_state' : trial.suggest_int('random_state', 1, 10000)
#     }
#     # 학습 모델 생성
#     model = CatBoostRegressor(**param)
#     model = model.fit(x_train, y_train) # 학습 진행
    
#     # 모델 성능 확인
#     y_predict = model.predict(x_test)
#     score = r2_score(y_test, y_predict)
    
#     print('score : ', score)
#     return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveCATBOOST, n_trials=300)

# best_params = study.best_params
# print("BEST PARAM: ",best_params) 


#3. 훈련


xgb_params=  {'n_estimators': 1973, 'max_depth': 8, 'min_child_weight': 10, 'gamma': 2.6161556938062667, 
              'learning_rate': 0.0032711508067253814, 'colsample_bytree': 0.43588430169744163,
              'lambda': 4.109502820985298, 'alpha': 1.094236763660126, 'subsample': 0.6818116013889788, 'n_jobs': -1}

lgbm_params=  {'n_estimators': 820, 'max_depth': 13, 'min_child_weight': 1,#'gamma': 0.8787397106554804,
            'learning_rate': 0.008262704782718127, 'colsample_bytree': 0.9525731858721108, 'num_leaves': 44,
            'alpha': 1.5559896871264183, 'subsample': 0.7299426393373126, 'n_jobs': -1, 'force_row_wise' : 'true', 'verbose' : 1}
cat_params= {'n_estimators': 1174, 'max_depth': 14, 'learning_rate': 0.030137373447938796, 'bootstrap_type': 'Bernoulli', 'boosting_type': 'Ordered'}

xgb = XGBRegressor(**xgb_params)
lgb = LGBMRegressor(**lgbm_params)
rf = RandomForestRegressor()
cb = CatBoostRegressor(**cat_params)
model = StackingRegressor(
     estimators=[('LGBM',lgb),
                 # ('RF',rf),
                 ('XGB',xgb)],
    final_estimator=CatBoostRegressor(verbose=0, **cat_params),
    #n_jobs= -1,
    cv=7, 
)
                    
#3. 훈련

model.fit(x_train, y_train)

#4.  평가, 예측
                    
 
print("=====================================")
y_submit = model.predict(test_csv)
submission_csv['Income'] = y_submit


r2 = model.score(x_test,y_test)
pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(pred,y_test))
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)

import datetime
dt = datetime.datetime.now()
y_submit = model.predict(test_csv)
submission_csv['Income'] = y_submit
submission_csv.to_csv(path+f'submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{rmse:4f}.csv',index=False)
print("R2:   ",r2)
print("RMSE: ",rmse)


pickle.dump(model, open(path + f'submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{rmse:4f}.dat', 'wb'))
# RMSE:  562.0450116401987

# cv 5
# R2:    0.3078894273354884
# RMSE:  556.721173816561


#cv 7
# R2:    0.3076517751190555
# RMSE:  556.8167471761383
