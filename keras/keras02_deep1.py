from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(200, input_dim=1)) #dense 레이어 한층을 추가한다는 뜻.
model.add(Dense(1024)) #위 레이어의 인풋과 밑 레이어 아웃풋이 똑같을때는 
model.add(Dense(512)) #노드 갯수, 레이어 깊이 조절 가능 훈련 취소 Ctrl C
model.add(Dense(206))
model.add(Dense(103))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측 값 : ", result)

#로스 :  1.4389213902177289e-05
#1/1 [==============================] - 0s 73ms/step
#4의 예측 값 :  [[4.0000453]]