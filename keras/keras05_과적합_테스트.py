import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10]) 

y = np.array([1,2,3,4,6,5,7,8,9,10])

#2. 모델구성

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x, y, epochs=1000, batch_size= 1)

#4. 평가, 예측
loss= model.evaluate(x, y)
print("로스 :", loss)
result= model.predict([11000,7])
print("11000의 예측값: ", result)

#로스 : 1.287503415513258e-12
#1/1 [==============================] - 0s 48ms/step
#11000의 예측값:  [[1.0999998e+04] e+는 소수점 추가
# [6.9999986e+00]]
