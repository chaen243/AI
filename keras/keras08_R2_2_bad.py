# 고의적으로 R2값 낮추기.
# 1. R2를 음수가 아닌 0에 가깝게 만들기.
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋과 아웃풋 포함해서 7개 이상.
# 4. batch_size = 1
# 5. 히든레이어의 노드는 10개 이상 100개 이하
# 6. train 75%
# 7. epoch 100번 이상

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8 ,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 90 ) #90

print(x_train)          
print(y_train)
print(x_test)
print(x_test)


#2. 모델구성

model = Sequential()
model.add(Dense(10, input_dim =1))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(60))
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

#로스 : 22.49988555908203
#R2 스코어 : 0.013162892198760878
#test_size= 0.25, random_state= 90 에포 200


