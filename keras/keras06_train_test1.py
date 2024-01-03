from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터

#x = np.array([1,2,3,4,5,6,7,8,9,10])

#y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7]) #(train=트레이닝)
y_train = np.array([1,2,3,4,6,5,7]) # 60~90%데이터 훈련. 나머지로 테스트

x_test = np.array([8,9,10]) #신뢰판단의 기준이 더 높기때문에 훈련데이터와 평가 데이터를 나눔
y_test = np.array([8,9,10])

#2. 모델구성

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=1000, batch_size= 1)

#4. 평가, 예측
loss= model.evaluate(x_test, y_test) #^트레이닝한 값의 로스가 통상적으로 더 적다.
print("로스 :", loss)
result= model.predict([11000,7])
print("11000의 예측값: ", result)

#로스 : 0.0015349858440458775
#1/1 [==============================] - 0s 58ms/step
#11000의 예측값:  [[1.0848278e+04]