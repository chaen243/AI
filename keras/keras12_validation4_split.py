from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터

x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.85, shuffle = True, random_state= 34)

print(x_train, y_train)
print(x_test, y_test)

#2. 모델구성

model = Sequential()
model.add(Dense(4, input_dim = 1))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 1000, batch_size= 1, validation_split= 0.35, verbose=1) #val 데이터를 0.4만큼 쓴다.

#4. 평가, 예측
loss= model.evaluate(x_test, y_test) #evaluate>val>loss 순으로 신뢰도^
print("로스 :", loss)
result= model.predict([11000,17])
print("11000의 예측값: ", result)