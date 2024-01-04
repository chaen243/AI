import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])

y = np.array([1,2,3,4,6,5,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾기

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 405 ) #0.01
                                    #^기본적으로 x, y를 제외한 값이 들어가 있지만 훈련할때마다 랜덤으로 돌아가게됨
                                    #random_state 랜덤난수가 존재. 랜덤난수의 규칙대로 나감.
                                    #shuffle = true 디폴트. false로 바꾸면 순서대로 자름
                                    #train_size = 75%(0.75) 디폴트.
                                    #train_size, test_size는 1.0 이하로 맞춰야함 1.0이상 올라가면 에러. 1.0 미만이면 데이터 손실은 있지만 훈련 가능.


print(x_train) #()안에 들어가는거 ==파라미터(매개변수)
print(y_train)
print(x_test)
print(y_test)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(30))
model.add(Dense(54))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1 )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)
result = model.predict([x])
print("11의 예측값:", result)



import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.plot(x, result, color= 'red')
plt.show()