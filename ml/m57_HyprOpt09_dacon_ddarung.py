#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
import time


#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"
# print(path + "aaa.csv") 문자그대로 보여줌 c:\_data\daicon\ddarung\aaa.csv
# pandas에서 1차원- Series, 2차원이상은 DataFrame이라고 함.

train_csv = pd.read_csv(path + "train.csv", index_col=['id']) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
#submission_csv = pd.read_csv(path + "submission.csv") 
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


print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= False, random_state= 6) #399 #1048 #6


scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델구성
from bayes_opt import BayesianOptimization
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import warnings
warnings.filterwarnings('ignore')


search_space = {
    'learning_rate' : hp.uniform('learning_rate',0.001, 0.1),
    'max_depth' : hp.quniform('max_depth', 3, 10, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 40, 1),
    'min_child_samples' : hp.quniform('min_child_samples',10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight',1, 50, 1),
    'subsample' : hp.uniform('subsample',0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree',0.5, 1),
    'max_bin' : hp.quniform('max_bin',9, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda',-0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha',0.01, 50)
}


def xgb_hamsu(search_space):
    #정수형(반올림())
    params = {
        'n_estimators' : 100,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        # SUBSAMPLE = 0~1사이. 최대값(최소값(subsample)1)0)
        # min1 ~ max0 사이 무조건 0~1사이값이 나옴.
        'subsample' : max(min(search_space['subsample'], 1), 0),
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']), 10),
        'reg_lambda' : max(search_space['reg_lambda'], 0),
        'reg_alpha' : search_space['reg_alpha']
    }
    model = XGBRegressor(**params, n_jobs = -1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric = 'mae',
              verbose = 0,
              early_stopping_rounds=50,
    )
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    if 'best_score' not in xgb_hamsu.__dict__:
        xgb_hamsu.best_score = results
        print(f"R2 : {results}")
    return results

trial_val = Trials()
n_iter = 500

start_time = time.time()
best = fmin(
    fn = xgb_hamsu,
    space= search_space,
    algo= tpe.suggest,
    max_evals=50,
    trials= trial_val,
    rstate=np.random.default_rng(seed=10),
)
end_time = time.time()

print('best :', best)

print(n_iter, '번 걸린시간 :', round(end_time - start_time, 2), '초')

# 일반모델 r2 : 0.7599832100915234
# PF r2 : 0.764123931434631

# best : {'colsample_bytree': 0.8647398255924487, 'learning_rate': 0.0014989643637105397, 
#         'max_bin': 145.0, 'max_depth': 9.0, 'min_child_samples': 104.0, 'min_child_weight': 38.0, 
#         'num_leaves': 24.0, 'reg_alpha': 27.680948409372547, 'reg_lambda': 2.2337563125222974, 'subsample': 0.9639773645054783}
# 500 번 걸린시간 : 3.62 초
# R2 : 0.3319639067870035