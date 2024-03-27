import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
import time
import random

random.seed(42)
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)


train_csv = pd.read_csv('D:\hyper/train.csv')

# person_id 컬럼 제거
x = train_csv.drop(['person_id', 'login'], axis=1)
y = train_csv['login']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state= 42)




def objectiveRF(trial):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 10, 1000),
        'criterion' : trial.suggest_categorical('criterion', ['gini','entropy']),
        'bootstrap' : trial.suggest_categorical('bootstrap', [True,False]),
        'max_depth' : trial.suggest_int('max_depth', 2, 35),
        'random_state' : 42,
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 200),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 200),
        # 'min_samples_split' : trial.suggest_uniform('min_samples_split',0,1),
        # 'min_samples_leaf' : trial.suggest_uniform('min_samples_leaf',0,1),
        'min_weight_fraction_leaf' : trial.suggest_float('min_weight_fraction_leaf',0, 0.5),
    }
            
    # RandomForestClassifier 객체 생성
    model = RandomForestClassifier(**param)
    rf_model = model.fit(x_train, y_train)
    
    score = model.score(x_test, y_test)
    
    return score


while True:
    study = optuna.create_study(direction='maximize')
    study.optimize(objectiveRF, n_trials=5000)
    
    best_params = study.best_params
    
    model = RandomForestClassifier(**best_params)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    pred = model.predict_proba(x_test)[:,1]
    print("score : ", score)
    auc = roc_auc_score(y_test,pred)
    print("auc", auc)

    submit = pd.read_csv('D:\hyper/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
    for label in submit:
        if label in best_params.keys():
            submit[label] = best_params[label]
    import datetime
    dt = datetime.datetime.now()

    # submit.to_csv('D:\hyper/baseline_submit2.csv', index=False)
    submit.to_csv(f'c:\\hyper\\146_submit_{dt.day}day{dt.hour:2}{dt.minute:2}_AUC_{auc:.6f}.csv',index=False)
    print("최적의 파라미터 : ", best_params)
