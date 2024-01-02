# from tensorflow.keras.models import Sequential  # Ctrl / = 주석처리, 라인 앞에서 Ctrl C 누르면 라인 전체 복사 Shift del 한줄 삭제
# from tensorflow.keras.layers import Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()#모델의 변수명을 정의. 순차적모델을 만듬
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x, y, epochs=10000)

#4. 평가, 예측
loss= model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([7])
print("7의 예측값 : ", result) #loss 핑퐁 직전 구간 찾아내기!

# 로스 :  0.32380953431129456
# 1/1 [==============================] - 0s 40ms/step
# 7의 예측값 :  [[6.799998]]


# 로스 :  0.32380950450897217
# 1/1 [==============================] - 0s 49ms/step
# 7의 예측값 :  [[1.1428598]
#  [2.085716 ]
#  [3.028572 ]
#  [3.9714282]
#  [4.914284 ]
#  [5.8571405]
#  [6.7999964]]
# 에포 : 10000