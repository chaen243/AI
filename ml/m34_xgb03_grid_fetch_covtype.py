import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
from keras.utils import to_categorical
import time
from sklearn.metrics import accuracy_score
import time
#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = (datasets.target)-1
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor




n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10], "gamma": [0,2,4]},
    {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10],"gamma": [0,2,4]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10],"gamma": [0,2,4]},
    {"min_samples_split": [2, 3, 5, 10], "subsample": [0, 0.3, 1],"gamma": [0,2,4]},
    {"reg_alpha": [0.2351, 0.135], "min_samples_split": [2, 3, 5, 10], "max_depth" : [4, 8, 10, 11]}]





model = RandomizedSearchCV(XGBClassifier(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     refit= True, #디폴트 트루~
                     random_state= 123,
                     )#n_jobs=-1) #CPU 다 쓴다!




#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)                #train의 결과이기 때문에 절대적으로 믿을 수 는 없음.

results = model.score(x_test, y_test)

print('score :', results)

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")
#model.score : 0.7131677987883238

#Linear
# model.score : 0.5243826877180099
# acc : 0.5243826877180099
# 걸린시간 : 348.6212000846863 초

# KNeighborsClassifier
# acc : [0.96832268 0.96858085 0.96873548 0.96958744 0.96873548] 
#  평균 acc : 0.9688

#Stratified
# acc : [0.96861527 0.96858945 0.96864942 0.96855476 0.96852894] 
#  평균 acc : 0.9686

# cross_val_predict ACC : 0.881947970362211

# 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score : 0.9503559505068866
# score : 0.948779291412442
# accuracy_score : 0.948779291412442
# 최적튠 ACC : 0.948779291412442
# 걸린시간 : 1983.18 초