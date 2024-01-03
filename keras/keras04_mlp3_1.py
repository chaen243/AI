import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
#pip install numpy(텐써플로 설치할때 자동설치)

#1. 데이터
x= np.array([range(1, 10)]) #[[1 2 3 4 5 6 7 8 9]]
print(x)                    #(1, 9)
print(x.shape)                    
#^레인지에 대한 설명을 위한것


x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape)

x=x.T
print(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])  #[1,2...]-(두개이상은)list. (파이썬 안에서)

y=y.T
# 예측 : [10, 31, 211]

#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim = 3))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(150))
model.add(Dense(30))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x, y, epochs= 1000, batch_size= 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 :", loss)
result = model.predict([[10, 31, 211]])
print("10,31,21의 예측값 :", result)

#로스 : 1.1060308051369372e-11
#1/1 [==============================] - 0s 71ms/step
#10,31,21의 예측값 : [[11.000009   1.9999957]]