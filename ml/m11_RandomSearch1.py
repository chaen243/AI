import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

#1. 데이터

x,y = load_iris(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},    # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001,0.0001]},       # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                      # 24
     "gamma":[0.01,0.001, 0.0001], "degree":[3, 4]}                     
]# kernel별로 훈련해서 횟수를 줄여줌.

#2. 모델

#model = SVC(C=1, kernel='linear', degree=3)

# model = GridSearchCV(SVC(), 
#                      parameters, 
#                      cv=kfold, 
#                      verbose=1, 
#                      refit= True, #디폴트 트루~
#                      )#n_jobs=-1) #CPU 다 쓴다!



model = RandomizedSearchCV(SVC(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     refit= True, #디폴트 트루~
                     random_state=42,
                     n_iter=10,
                     n_jobs=-1) #CPU 다 쓴다!


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
#최적의 매개변수 :  SVC(C=1, kernel='linear') #지정한것을 제외하고도 가장 좋은것!
print("최적의 파라미터 : ", model.best_params_)
#최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 지정한것들중 가장 좋은것
print('best_score :', model.best_score_)
# fit 한것의 best_score : 0.975
print('score :', model.score(x_test, y_test))
# evaluate score : 0.9666666666666667

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

import pandas as pd
print(pd.DataFrame(model.cv_results_))






