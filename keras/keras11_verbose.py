#06_1 카피.

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터

#x = np.array([1,2,3,4,5,6,7,8,9,10]) 

#y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7]) #(train=트레이닝)
y_train = np.array([1,2,3,4,6,5,7]) # x데이터의 60~90%데이터 훈련. 나머지로 테스트 돌림.

x_test = np.array([8,9,10]) # 신뢰도 판단의 기준이 더 높기때문에 훈련데이터와 평가 데이터를 나눔 
y_test = np.array([8,9,10]) # 전체 데이터로 훈련을 돌리면 과적합의 문제,확률이 많아짐.
                            # test 데이터는 가중치에 영향을 주지않음.
#2. 모델구성                # 통상적 실무 데이터에서는 일부를 빼지만 데이터 범위 유지를 위해 전체데이터에서 임의로 뽑아내야함.

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=1000, batch_size= 1, 
          verbose=100) 
#verbose=0 :침묵 에포 돌아가는 화면 보여주지 않아서 훈련 속도가 더 빨라짐.
#verbose=1 :디폴트
#verbose=2 : 프로그래스바 삭제 (큰데이터 돌릴땐 디폴트값이 나음)
#verbose=3 : 에포만 나옴.
#verbose=나머지숫자 : 에포만 나온다.


#4. 평가, 예측
loss= model.evaluate(x_test, y_test) #^트레이닝한 값의 로스가 통상적으로 더 적게 나옴.
print("로스 :", loss)
result= model.predict([11000,7])
print("11000의 예측값: ", result)

#로스 : 0.0015349858440458775
#1/1 [==============================] - 0s 58ms/step
#11000의 예측값:  [[1.0848278e+04]