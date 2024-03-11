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
import warnings
warnings.filterwarnings('ignore')


bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    #정수형(반올림())
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        # SUBSAMPLE = 0~1사이. 최대값(최소값(subsample)1)0)
        # min1 ~ max0 사이 무조건 0~1사이값이 나옴.
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),
        'reg_lambda' : max(reg_lambda, 0),
        'reg_alpha' : reg_alpha
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
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds = bayesian_params,
    random_state = 777
)

n_iter = 500
start_time = time.time()
bay.maximize(init_points=5, n_iter = n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time - start_time, 2), '초')


# 일반모델 acc : 0.98
# PF acc : 0.9822222222222222



# {'target': 0.9822222222222222, 'params': {'colsample_bytree': 0.5, 'learning_rate': 1.0, 'max_bin': 195.73059431381643, 'max_depth': 10.0, 'min_child_samples': 144.10198503584067, 'min_child_weight': 1.0, 'num_leaves': 35.61488547908101, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 1.0}}
# 500 번 걸린시간 : 264.1 초