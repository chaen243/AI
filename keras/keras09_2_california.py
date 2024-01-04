import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 282 )

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

#2. 모델구성

model = Sequential()
model.add(Dense(8,input_dim = 8))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs= 5000, batch_size= 200)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)
y_predict = model.predict(x_test) 
result = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)

print("걸린 시간 :", round(end_time - start_time, 2), "초")



#로스 : 0.5067346692085266
#R2 스코어 : 0.6165943551579782
#epochs 5000, batch_size= 200
#8, 16, 10, 8, 4, 1 랜덤 282