from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

datasets = load_digits() #mnist 원판
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape) #(1797, 64)
print(y.shape) #(1797,)
print(pd.value_counts(y, sort= False)) #sort= False 제일 앞 데이터부터 순서대로 나옴
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, stratify=y)


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

# 일반모델 acc : 0.98
# PF acc : 0.9822222222222222




# acc : 0.7555555555555555
# best : {'colsample_bytree': 0.9111141836542521, 'learning_rate': 0.05494333773498434, 'max_bin': 239.0, 
#         'max_depth': 10.0, 'min_child_samples': 193.0, 'min_child_weight': 50.0, 'num_leaves': 24.0, 
#         'reg_alpha': 48.99402607647728, 'reg_lambda': 3.241937849138009, 'subsample': 0.5078067012142962}
# 500 번 걸린시간 : 11.28 초