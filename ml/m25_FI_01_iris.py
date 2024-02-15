#keras18 copy
#레거시한 머신러닝~~

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC #softvector machine
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'
    
aaa = CustomXGBClassifier()
# print(aaa)   #XGBClassifier()
#두가지 방법!


#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target    
print(datasets.feature_names)   

    
# ####넘파이####    
# x = np.delete(x, 0, axis=1)   
# print(x)
    
####판다스로 바꿔서 컬럼 삭제 ########   
x = pd.DataFrame(x, columns = datasets.feature_names)
x = x.drop('sepal length (cm)', axis=1)
print(x)
 

     

#실습
#피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성후 기존 모델결과와 비교    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  shuffle= True, random_state= 123, stratify= y)



#2. 모델구성


models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), CustomXGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print('=====================')
    print(model, "acc :", acc)
    print(model,':', model.feature_importances_)
        
'''     

# #model = DecisionTreeClassifier(random_state=777)
# # model = RandomForestClassifier(random_state=777)
# # model = GradientBoostingClassifier(random_state=777)
# model = XGBClassifier()



# #3. 컴파일, 훈련



# model.fit(x_train, y_train)

# #4. 평가, 예측

# results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
# print("model.score: ", results)

# y_predict = model.predict(x_test)
# print(y_predict)



# acc = accuracy_score(y_predict, y_test)
# print(model, "acc :", acc)

# print(model, ':', model.feature_importances_)
#[0.01666667 0.01666667 0.0607589  0.90590776] #어떤 피쳐가 중요한지에 대한 중요도 (트리계열 모델에서만 제공해줌.)


# DecisionTreeClassifier acc : 0.8333333333333334
# DecisionTreeClassifier : [0.         0.0425     0.92133357 0.03616643]
# RandomForestClassifier acc : 0.9333333333333333
# RandomForestClassifier : [0.09394744 0.02607627 0.45664644 0.42332985]
# GradientBoostingClassifier acc : 0.9666666666666667
# GradientBoostingClassifier : [0.0016088  0.02142657 0.76742254 0.2095421 ]
# XGBClassifier acc : 0.9333333333333333
# XGBClassifier : [0.02430454 0.02472077 0.7376847  0.21328996]

#drop
# =====================
# DecisionTreeClassifier() acc : 0.8333333333333334
# DecisionTreeClassifier() : [0.0425     0.42133357 0.53616643]
# =====================
# RandomForestClassifier() acc : 0.9333333333333333
# RandomForestClassifier() : [0.13540505 0.39992492 0.46467003]
# =====================
# GradientBoostingClassifier() acc : 0.9666666666666667
# GradientBoostingClassifier() : [0.0213324  0.74683199 0.23183561]
# =====================
# XGBClassifier() acc : 0.9333333333333333
# XGBClassifier() : [0.02697425 0.84159774 0.131428  ]
'''