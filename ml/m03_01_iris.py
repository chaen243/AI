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
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#1. 데이터

X, y = load_iris(return_X_y=True)
print(X.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size= 0.8,  shuffle= True, random_state= 11, stratify= y)



#2. 모델구성

#model = LinearSVC()
#C가 크면 training포인트를 정확히 구분(굴곡지다), c가 작으면 직선에가깝다.
#model = Perceptron()
model = LogisticRegression() #model.score:  0.7666666666666667
model = KNeighborsClassifier()
model = DecisionTreeClassifier()
model = RandomForestClassifier()


#3. 컴파일, 훈련



model.fit(x_train, y_train)

#4. 평가, 예측

results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
print("model.score: ", results)

y_predict = model.predict(x_test)
print(y_predict)



acc = accuracy_score(y_predict, y_test)
print("acc :", acc)


