import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터

x = np.array([[[1,2,3,4,5,6],
             [2,3,4,5,6,7],
             [0.1,0.2,0.3,0.4,0.5,0.6]]])

y = np.array([1,2,3,4,5,6])

x = x.T

#2. 모델구성

model = Sequential()
model.add(Dense(6, input_dim = 3))
model.add(Dense(78))
model.add(Dense(120))
model.add(Dense(40))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer = 'adam')
model.fit(x, y, epochs= 100, batch_size= 2)

#4. 평가, 예측
loss = (model.evaluate(x, y))
print("로스 :", loss)
result = model.predict([[6,7,0.6]])
print("[6,7,0.6]의 예측값 : ",result)
          
#로스 : 6.975634653239027e-11
#1/1 [==============================] - 0s 67ms/step
#[6,7,0.6]의 예측값 :  [[5.9999957]]          