import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
#pip install numpy(텐써플로 설치할때 자동설치)

#1. 데이터
x= np.array([range(1, 10)]) #[[1 2 3 4 5 6 7 8 9]]
print(x)                    #(1, 9)
print(x.shape)                    
#^레인지에 대한 설명을 위한것


x = np.array([range(10)])
x=x.T

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]])

y=y.T

#2. 모델구성

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(3))

#3. 컴파일,훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x, y, epochs=500, batch_size= 1)

#4. 평가, 예측
loss= model.evaluate(x, y)
print("로스 :", loss)
result= model.predict([10])
print("10의 예측값: ", result)

#로스 : 1.3227935999164664e-13
#1/1 [==============================] - 0s 61ms/step
#10의 예측값:  [[11.000002   1.9999995 -1.0000007]]
#배치1 에포 1300