from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time


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

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1],"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1], "min_samples_split": [2, 3, 5, 10]},
]    

# model = RandomForestClassifier()

model = GridSearchCV(RandomForestClassifier(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     refit= True, #디폴트 트루~
                     )#n_jobs=-1) #CPU 다 쓴다!



# model = RandomizedSearchCV(RandomForestClassifier(), 
#                      parameters, 
#                      cv=kfold, 
#                      verbose=1, 
#                      refit= True, #디폴트 트루~
#                      n_jobs=-1) #CPU 다 쓴다!

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

y_predict = model.predict(x_test)
print("acc_score", accuracy_score(y_test, y_predict))

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

#randomforest
# acc_score 0.9866666666666667
# 걸린시간 : 0.14 초

#RandomsearchCV
# acc_score 0.98
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 3}
# best_score : 0.965857083849649
# score : 0.98
# 최적튠 ACC : 0.98
# 걸린시간 : 1.74 초

#GridSearchCV
# acc_score 0.9866666666666667
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': -1}
# best_score : 0.9725237505163156
# score : 0.9866666666666667
# 최적튠 ACC : 0.9866666666666667
# 걸린시간 : 17.16 초