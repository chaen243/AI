import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

mms = MinMaxScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)


#2. 모델구성
 

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
              eval_metric = 'logloss',
              verbose = 0,
              early_stopping_rounds=50,
    )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
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

# best : {'colsample_bytree': 0.7470728059084097, 'learning_rate': 0.003743385343536733, 'max_bin': 332.0, 
#         'max_depth': 5.0, 'min_child_samples': 90.0, 'min_child_weight': 32.0, 'num_leaves': 29.0, 
#         'reg_alpha': 43.484203349687064, 'reg_lambda': 3.8027284538270587, 'subsample': 0.9299459818100199}
# 500 번 걸린시간 : 1.89 초