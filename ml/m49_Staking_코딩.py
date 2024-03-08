#배깅 보팅  부스팅 스태킹

'''
배깅- randomforest
모델이 1종류. 사용하는 데이터가 다르다. 
트리마다 데이터를 랜덤하게 가져와 샘플링을 다르게 하여 훈련함. 랜덤한 데이터중 중복해서 가져오기 가능.
(n_estimator) 디시전트리를 한번 훈련시켰다. 에포와 비슷!



보팅- 투표 (모델 여러개중 좋은거/평균값 사용하겠다)
hard - 단순하게 값이 좋은게 많은 것으로 사용. 다수결. (주로 모델갯수를 홀수로 사용)
soft - 값을 평균내서 그중 제일 좋은 모델을 채택.

스태킹- 훈련을 해서 나온 predict 값을 가지고 재훈련을 시킨다.
predict 값을 재훈련 시킬때는 얘가 x값이 된다.



부스팅- predict 오차가 많은 데이터에 가중치를 더 줘서 다음 다음 훈련에서 그 데이터를 더 집중할 수 있게 함.
그렇게 분류가 제대로 될 수 있게 해줌.
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify= y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Params = {'n_estimators' : 1000,
#           'learning_rate' : 0.01,
#           'max_depth' : 3,
#           'gamma' : 0,
#           'min_child_weight' : 0,
#           'subsample' : 0.4,
#           'colsample_bytree' : 0.8,
#           'colsample_lylevel' : 0.7,
#           'colsample_bynode' : 1,
#           'reg_alpha' : 0,
#           'reg_lamba' : 1,
#           'random_state' : 3377,
#           'verbose' : 0        
#           }

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model_class = [xgb, rf, lr]
li = []
li2 = []
for model in model_class:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    score = accuracy_score(y_test, y_pred_test)
    class_name = model.__class__.__name__
    print("{0} 정확도 : {1:4f}".format(class_name, score))
    li.append(y_pred)
    li2.append(y_pred_test)
   
new_x_train = np.array(li).T 
new_x_test = np.array(li2).T 

# print(new_x_train.shape)  #(455, 3)
# print(new_x_test.shape)  #(114, 3)

model2 = CatBoostClassifier(verbose=0)
model2.fit(new_x_train, y_train)
y_pred = model2.predict(new_x_test)
score2 = accuracy_score(y_test, y_pred)
print('최종 :', score2)



#결과물 - xgbclassifier ACC : 0.000
#randomforest acc : 000
# LogisticRegression : 000
# 스태킹 결과 : 0.000


# XGBClassifier 정확도 : 0.991228
# RandomForestClassifier 정확도 : 0.973684
# LogisticRegression 정확도 : 0.973684