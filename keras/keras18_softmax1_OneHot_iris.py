import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#1. 데이터

datasets= load_iris()
#print(datasets)
print(datasets.DESCR)

print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,) 데이터 확인 필수!클래스, 개별클래스 갯수 확인 꼭 할것 모델분류 잘못 할 확률도 있음.
print(y)
print(np.unique(y, return_counts= True)) #(array([0, 1, 2]), array([50, 50, 50],
print(pd.value_counts(y))
#0    50  학습시키기 좋게 하기 위해 같은 비율로 나눠놓음. 
#1    50  비율이 다를땐 학습이 어려움 (많이 차이가 나면 쏠림현상이 꼭 생김. 임의로 과적합이 잡힐 수 있다.)
#2    50  비율이 다를땐 증폭으로 라벨의 갯수를 비슷하게 맞춰줘야함. (성능 많이 올라감!) 데이터가 많아야 훈련의 결과가 좋음.

#판다스
# y = pd.get_dummies(y)

#케라스 (불리언데이터타입)
#y = to_categorical(y) 


#reshape (데이터의 위치,순서,갯수,내용을 바꾸지 않는다면 어떻게 나눠도 상관없음.) *중.요*
y = y.reshape(-1,1)
#y = y.reshape(-1,1) 
#[[1,2,3],[4,5,6]] (2,3) == [1,2,3,4,5,6] (6, ) == [[1,2],[3,4],[5,6]](3,2)






print("======= 원핫 3 ========")
#사이킷런 
#(친구들은 임포트, (클래스상태이기때문에)정의, 핏(훈련), 트랜스폼 의 형태를 사용) 외워!
ohe = OneHotEncoder(sparse= False)   #= to.array()노상관

#ohe = OneHotEncoder(sparse= True)  #디폴트 트루
#y = ohe.fit_transform(y).toarray() 
#ohe.fit(y)
#y = ohe.transform(y) == y와 다른친구들이 같이 나올땐 이 문법을 사용해야함
y = ohe.fit_transform(y) #위아래 같아여 fit + transform 대신 씀.






x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8,  shuffle= True, random_state= 11, stratify= y) #y의 라벨값을 비율대로 잡아줌 #회귀모델에서는 ㄴㄴ 분류에서만 가능
#print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
#print(y_train.shape, y_test.shape) #(7620, ) (3266, )
print(np.unique(y_test, return_counts = True ))



#2. 모델구성

model = Sequential()
model.add(Dense(100, input_dim = 4))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(3, activation = 'softmax'))



#3. 컴파일, 훈련
model.compile  (loss = 'categorical_crossentropy', optimizer = 'adam', metrics= 'acc')

es = EarlyStopping(monitor= 'val_loss', mode= 'min',
                   patience=100, verbose=2, restore_best_weights= True) #es는 verbose2가 es 정보를 보여줌.
start_time = time.time()
his = model.fit(x_train, y_train, epochs= 300, batch_size=1, validation_split= 0.25, verbose=2, callbacks= [es] ) #검증모델은 간접적인 영향을 미침.
end_time = time.time()
  

#4. 평가, 예측



results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("정확도 :", results[1])

y_predict = model.predict(x_test)





print(y_predict)

print(y_predict.shape, y_test.shape) #(30, 3) (30, 3)

y_test = np.argmax(y_test, axis=1) #원핫을 통과해서 아그맥스를 다시 통과시켜야함
y_predict = np.argmax(y_predict, axis=1 )
print(y_test, y_predict)
result = accuracy_score(y_test, y_predict)

acc = accuracy_score(y_predict, y_test)
print("acc :", acc)


print("걸린 시간 :", round(end_time - start_time, 2), "초" )
