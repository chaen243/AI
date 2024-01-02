# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense #단어만 카피할때는 클릭해서 카피
import numpy as np #자동완성 확인은 ctrl space
import tensorflow as tf
import keras
print("tf 버전 : ", tf.__version__)
print("keras 버전 :", keras.__version__)

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6]) 

#2.  모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # 노드의 갯수가 갑자기 줄어들면 값이 이상해질수 있음.
model.add(Dense(8)) #hidden layer 중간층에서 통상적으로 1은 잘 쓰지않는다.
model.add(Dense(120))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일,훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x, y, epochs=500, batch_size= 2)# batch_size =1 = 데이터를 하나씩 잘라서 훈련하겠다  #데이터를 쪼개서 작업하는것 
                                         # batch를 자르면 그만큼 연산량이 늘어남. 일괄처리할 데이터를 정하는것.
                                         #3 batchsize보다 작은 데이터면 size에 맞게 돌림
#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([7])
print("로스 : ", loss)
print("7의 예측값 :", result)

#로스 :  0.323810875415802
#7의 예측값 : [[6.7993507]]
#batch1, 1,9,90,4,1 에포- 500