#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, RandomForestRegressor, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
import time


#1. 데이터

x,y = load_diabetes(return_X_y=True)



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle= True)#, stratify=y)



# scaler = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



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
              eval_metric = 'logloss',
              verbose = 0,
              early_stopping_rounds=50,
    )
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
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



# best : {'colsample_bytree': 0.9964358418839089, 'learning_rate': 0.001035354513728975, 'max_bin': 372.0, 'max_depth': 4.0, 
#         'min_child_samples': 122.0, 'min_child_weight': 5.0, 'num_leaves': 24.0, 'reg_alpha': 25.3616160183033, 
#         'reg_lambda': 3.0337991548901346, 'subsample': 0.7164776432769465}
# 500 번 걸린시간 : 2.63 초
# r2 : 0.19490585601522237

