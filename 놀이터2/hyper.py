import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time
import random

random.seed(42)


train_csv = pd.read_csv('D:\hyper/train.csv')

# person_id 컬럼 제거
X_train = train_csv.drop(['person_id', 'login'], axis=1)
y_train = train_csv['login']

param_search_space = {
    'n_estimators': [995, 999, 998, 997, 996],
    'bootstrap': [True],
    'max_depth': [9, 8, 10, 6, 7],
    'min_samples_split': [2, 3, 4,],
    'min_samples_leaf': [1, 3, 2, 4, 5, 6],
    'min_weight_fraction_leaf': [0.00005,0.0005, 0.000005]
    
}
 
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) 
           
# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space,
                                  cv=kfold, n_jobs=-1, verbose=2, scoring='roc_auc',
                                  )

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_params, best_score

submit = pd.read_csv('D:\hyper/sample_submission.csv')


# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value
import datetime
dt = datetime.datetime.now()

# submit.to_csv('D:\hyper/baseline_submit2.csv', index=False)
submit.to_csv('D:\hyper/'+f"146_submit_{dt.day}day{dt.hour:2}{dt.minute:2}_.csv",index=False)
print("최적의 파라미터 : ", best_params)
print("score", best_score)
print('finish')
