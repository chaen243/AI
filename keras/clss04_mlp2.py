# [실습]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터

x = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0]
             ])                #제일 마지막 ','은 에러로 표시하지 않는다. 문제되지 않음.

y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(x.shape, y.shape) #두가지 다 보여줌. #(2,10) (10, )
x = x.T
print(x.shape) # (10, 2) #열의 갯수만큼 input_dim을 넣을 수 있다.

#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim = 3))
model.add(Dense(42))
model.add(Dense(90))
model.add(Dense(20))
model.add(Dense(1))



#3.컴파일, 훈련

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x, y, epochs=400 , batch_size= 1)
#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 :", loss)
result = model.predict([[10,1.3,0]])
print("[10,1.3,0]의 예측 값: ", result)

#로스 : 1.2457249676245397e-09
#1/1 [==============================] - 0s 67ms/step
#[10,1.3,0]의 예측 값:  [[10.000032]]
#에포=300 배치=1 5,42,90,24,1

#로스 : 6.190248400678167e-12
#1/1 [==============================] - 0s 59ms/step
#[10,1.3,0]의 예측 값:  [[10.000005]]
#에포=4000 배치=1 단층

#로스 : 0.0
#1/1 [==============================] - 0s 64ms/step
#[10,1.3,0]의 예측 값:  [[10.]]


#model.add(Dense(42))
#model.add(Dense(90))
#model.add(Dense(20))
#model.add(Dense(1))




