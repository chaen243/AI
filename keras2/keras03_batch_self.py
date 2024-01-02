import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터

x = np.array([2,3,4,5,6,7])
y = np.array([2,3,4,6,5,7])

#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim =1))
model.add(Dense(54))
model.add(Dense(120))
model.add(Dense(24))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs= 100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 ;", loss)
result = model.predict([8])
print("8의 예측값:", result)