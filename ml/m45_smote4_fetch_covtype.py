import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
import sklearn as sk
# print('사이킷런', sk.__version__)

smote = SMOTE(random_state=123)
x_train, y_train =smote.fit_resample(x_train, y_train)

print(pd.value_counts(y_train)) 

# 0    226640
# 1    226640
# 2    226640
# 5    226640
# 6    226640
# 4    226640
# 3    226640


# Params = {'n_estimators' : 1500,
#           'learning_rate' : 0.05,
#           'max_depth' : 10,
#           'gamma' : 0,
#           'min_child_weight' : 0,
#           'subsample' : 0.4,
#           'colsample_bytree' : 0.8,
#           'colsample_bynode' : 1,
#           'reg_alpha' : 0,
#           'random_state' : 1234,
#           }


# model = XGBClassifier(tree_method = 'hist', device = "cuda")
# model.set_params( **Params)


model = RandomForestClassifier()

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score

results = model.score(x_test, y_test)
print('score :', results)
y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))
print("걸린시간 :", round(end_time - start_time, 2), "초")
