from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time


#R2 0.62 이상
#1. 데이터
datasets = load_diabetes()
x = np.array(datasets.data)
y = np.array(datasets.target)

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 442 ) #442
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, shuffle=True, random_state= 442)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 10))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 100, batch_size = 5, validation_data= (x_val, y_val))
end_time = time.time()


#4. #4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)
y_predict = model.predict(x_test) 
result = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)

print("걸린 시간 :", round(end_time - start_time, 2), "초")

#로스 : 2265.661376953125

#R2 스코어 : 0.6538567472714809