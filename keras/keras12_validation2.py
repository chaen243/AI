import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split




#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

#1-10 트레인 11-13 발리 14-16 테스트

x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))

x_val = np.array(range(11, 14))
y_val = np.array(range(11, 14))

x_test = np.array(range(14, 17))
y_test = np.array(range(14, 17))

print(x_train)
print(x_val)
print(x_test)



#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs= 300, batch_size= 1, verbose = 1, validation_data= (x_val, y_val))

#4. 평가, 예측
loss= model.evaluate(x_test, y_test) #^트레이닝한 값의 로스가 통상적으로 더 적다.
print("로스 :", loss)
result= model.predict([11000,17])
print("11000의 예측값: ", result)