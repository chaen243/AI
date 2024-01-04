from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8 ,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 450 ) #450

print(x_train)          
print(y_train)
print(x_test)
print(x_test)


#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size= 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 예측값과 y test 실제값을 평가한것.
print("로스 :", loss)
y_predict = model.predict(x_test) #평가는 항상 테스트값 사용.
result = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)

import matplotlib.pyplot as plt

plt.scatter(x, y)
#plt.plot(x, result, color = 'red') plot 선
plt.scatter(x,result , color = 'red') #그래프로 나타내는것 = 시각화한다.

plt.show()

#로스 : 0.3103421926498413
#1/1 [==============================] - 0s 49ms/step
#1/1 [==============================] - 0s 16ms/step
#R2 스코어 : 0.9953155897660457
#test_size= 0.2, random_state= 450
