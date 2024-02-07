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


#1. 데이터

datasets= load_iris()
#print(datasets)
print(datasets.DESCR)

print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) 
print(y)
print(np.unique(y, return_counts= True))
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  shuffle= True, random_state= 11, stratify= y)
print(np.unique(y_test, return_counts = True ))



#2. 모델구성

# model = Sequential()
# model.add(Dense(100, input_dim = 4))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(5))
# model.add(Dense(3, activation = 'softmax'))
model = LinearSVC(C=110)
#C가 크면 training포인트를 정확히 구분(굴곡지다), c가 작으면 직선에가깝다.



#3. 컴파일, 훈련
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
#fit에 compile이 들어있음!

# es = EarlyStopping(monitor= 'val_loss', mode= 'min',
#                    patience=100, verbose=2, restore_best_weights= True) #es는 verbose2가 es 정보를 보여줌.


model.fit(x_train, y_train)

#4. 평가, 예측
#results = model.evaluate(x_test, y_test)

results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
print("model.score: ", results)

y_predict = model.predict(x_test)
print(y_predict)


# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1 )
# print(y_test, y_predict)
# result = accuracy_score(y_test, y_predict)

acc = accuracy_score(y_predict, y_test)
print("acc :", acc)


