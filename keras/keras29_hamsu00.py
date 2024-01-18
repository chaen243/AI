#keras04_mlp1 copy


import numpy as np  #multi layer perceptrone(노드)
from keras.models import Sequential, Model #함수형 모델
from keras.layers import Dense, Input
#오늘 과제. 텐서이상 20개 만들어서 풀어서 제출

#1. 데이터

x = np.array([[1,2,3,4,5,6,7,8,9,10], #열의 결측치 데이터는 Nan으로 메꿔서 갯수를 맞춰놓아야함.
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]#괄호 사이에',' 꼭 표시할것.[]사이의 수들은 하나의 데이터덩어리임.
             )

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)  # (2, 10)
print(y.shape)  # (10,) < 스칼라가 10개짜리 벡터가 하나라는 뜻 열(스칼라)의 갯수는 무조건 동일해야함.
x = x.T #행과 열을 바꾸는 기능 (데이터 정제의 개념)
# =x = x.transpose()
#[[1,1],[2,1,1],[3,1.2],...[10,1.3]]
print(x.shape)  # (10, 2) #shape 찍어보는거 버릇들이기!


# #2. 모델구성 (순차적)
# model = Sequential()
# model.add(Dense(10, input_shape = (2,))) #'열'= '컬럼' = 속성 = 특성 = '차원' #(행무시, 열우선)
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))


# model.summary()

#2. 모델구성 (함수형) 위아래 동일한 모델.
input1 = Input(shape=(2,))  #앞쪽 클래스 정의는 맘대로 가능
dense1 = Dense(10)(input1) #인풋값을 명시해주는것
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs= input1, outputs= output1)

model.summary()

# #3. 컴파일, 훈련
# model.compile(loss= 'mse', optimizer = 'adam')
# model.fit(x, y, epochs=100, batch_size= 2)

# #4.평가, 예측
# loss = model.evaluate(x, y)
# print("로스 :", loss) 
# results = model.predict([[10, 1.3]]) 
# print("[10, 1.3]의 예측값 :", results) 
# #[실습] : 소수점 2자리까지 맞추기
# #로스 : 0.03408418223261833
# #1/1 [==============================] - 0s 67ms/step
# #[10, 1.3]의 예측값 : [[10.002671]]