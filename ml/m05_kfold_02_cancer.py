import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold
from sklearn.svm import SVC

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
model = SVC()

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()


#2. 모델구성
model = SVC()




#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))



# acc : [0.92982456 0.92105263 0.92982456 0.92982456 0.87610619] 
#  평균 acc : 0.9173

#stratifiedKFold
# acc : [0.88596491 0.92982456 0.94736842 0.92105263 0.91150442] 
#  평균 acc : 0.9191



#LinearSVC
# model.sore: 0.9617486338797814
# acc : 0.9617486338797814