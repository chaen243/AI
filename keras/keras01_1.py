import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.             
print (tf.__version__)  # 2.15.0
from tensorflow.keras.models import Sequential # from안에 imoort를 땡겨옴 (순차적으로 땡겨옴) #warning 경고는 하지만 실행 가능, error 실행불가
from tensorflow.keras.layers import Dense     
import numpy as np 

#1. 데이터(정제,전처리) (데이터가 없으면 인공지능 학습불가)
x = np.array([1,2,3]) #넘파이 데이터형식의 1,2,3을 준비한다
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()    #모델의 변수명을 정의. 순차적모델을 만듬
model.add(Dense(1, input_dim=1)) #1=y(갯수), input_dim=1 (차원이 1개, x데이터가 1덩어리)

#3. 컴파일, 훈련 (기계어로 바꾸기, 최적의 weight 찾기)
model.compile(loss='mse', optimizer= 'adam') #loss 뒤 수식. loss값은 항상 양수. 0이 가장 낮음(가장 좋은 값). m(square)e= loss를 제곱하는 방식의 수식. 옵티/아담 현재로써 가장 좋은값. 컴파일은 행동하지않은 상태.
model.fit(x, y, epochs=8000) #fit= 훈련 훈련양 조절 필수. 과하게 훈련하면 과적합 걸릴 수 있음. #최적의 웨이트값이 생성

#4. 평가, 예측 (loss값 평가, 정확도등/)
loss = model.evaluate(x, y) #생성된 웨이트값을 빼줌
print("로스 : ", loss) #""안의 글자는 그대로 출력
result = model.predict([4]) #변수는 임의로 지정 가능.
print("4의 예측값 : ", result)


# 로스 :  2.4253192770079535e-12 (3000)
# 1/1 [==============================] - 0s 39ms/step
# 4의 예측값 :  [[4.000004]]

#평균 8000선에서 소수점 4자리 로스