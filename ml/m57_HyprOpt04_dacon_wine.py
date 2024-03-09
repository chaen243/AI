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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
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
    model = XGBClassifier(**params, n_jobs = -1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric = 'mlogloss',
              verbose = 0,
              early_stopping_rounds=50,
    )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    if 'best_score' not in xgb_hamsu.__dict__:
        xgb_hamsu.best_score = results
        print(f"acc : {results}")
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



# 일반모델 acc : 0.6781818181818182
# PF acc : 0.6981818181818182

# acc : 0.5454545454545454
# best : {'colsample_bytree': 0.9345485939069947, 'learning_rate': 0.013258956359146975, 
#         'max_bin': 184.0, 'max_depth': 9.0, 'min_child_samples': 36.0, 'min_child_weight': 37.0, 'num_leaves': 25.0,
#         'reg_alpha': 49.04918131136639, 'reg_lambda': 5.9500289026647515, 'subsample': 0.9976112809044521}
# 500 번 걸린시간 : 12.46 초