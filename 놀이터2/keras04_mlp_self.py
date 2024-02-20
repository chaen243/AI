import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[[1,2,3,4,5,6,7],
             [2,3,4,5,6,7,8],
             [1,2,3,4,5,6,7]]]
             )

y = np.array([1,2,3,4,5,6,7])

x = x.T

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim = 3))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=100, batch_size= 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 :", loss)
result = model.predict([[7,8,7]])
print("7의 예측값 :", result)

