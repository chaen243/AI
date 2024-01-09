import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split




#1. 데이터
x = np.arange(1, 17)
y = np.arange(1, 17)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, test_size=0.15, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)



#2. 모델구성

model= Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(4))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs= 100, batch_size= 1, validation_data= (x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result= model.predict([10000,16])
print("로스 :", loss)
print("10000,16의 예측 값:", result)
