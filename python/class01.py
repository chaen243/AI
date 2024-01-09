import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

#모든 인공지능 모델은 #1.데이터 #2모델구성 #3컴파일,훈련 #4 평가,예측의 순서로 함.
1데이터& 데이터의 전처리가 가장 중요.
model = Sequential() 모델의 변수명을 정의  (순차적인 모델이다. 라는 뜻)
input_dim = *** ***에는 열의 갯수가 들어감. 행무시 열우선!
loss = model.evaluate(x,y) 생성된 웨이트값을 빼줌.
reslt= model.predict([]) []안의 수를 예측. 괄호안의 변수는 임의로 지정 가능.
훈련횟수가 많다고해서 무조건 최적의 weight가 나오지 않음. 과적합 직전 구간까지 찾아내는 능력 키우기
batch_size = 32 디폴트값.  x개만큼 잘라서 훈련하는것. 작게 자를수록 연산량이 늘어남.
mlp = multi layer perceptrone(노드). 열의 갯수는 무조건 동일해야 훈련가능! 열/컬럼/속성/특성/차원!!
x.T = x.transpose 행과 열을 바꿔주는 기능.
range(10) = [0,1,2,3,4,5,6,7,8,9] <기계어에서는 숫자가 0부터 시작.
train/test- x데이터의 60~90%의 데이터로 훈련, 나머지로 테스트. test데이터는 가중치에 영향을 주지않음.
훈련 신뢰도 판단의 기준이 높아지기때문에 데이터 손실을 감수하더라도 훈련 데이터를 나누는것.
데이터 범위의 유지를 위해 전체 데이터에서 임의로 뽑음 (random_state)
x_train, x_test, y_train, y_test = train_test_split(x, y, test size= 0.75, shuffle = True, random_state = 0)
순서대로 적어야 데이터가 꼬이지 않음. random_state는 랜덤난수가 존재해 랜덤난수의 규칙대로 돌아감.
shuffle = True 에서 False로 바꾸면 데이터의 앞쪽부터 순서대로 끊어줌.

import matplotlib.pyplot as plt
plt.plot(x, result, color = 'red')
plt.show                             #실제값과 예측값을 그래프에 선으로 보여줌.

#R2 계산식
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

#in pandas
#1차원 - Series, 2차원이상 DataFrame.
train_csv = pd.read_csv(path + "train.csv", index_col=0) # train_csv는 path(경로) + train.csv 경로에 있고 그 0번째 컬럼은 인덱스 컬럼이다.
#컬럼명(header, id, 날짜등 통상적 첫번째 컬럼은 데이터가 아닌 인덱스컬럼일 확률이 높음. 열에서는 0번째 컬럼은 인덱스 처리가 디폴트값.)
x = train_csv.drop(['count'], axis=1) # x는 train_csv에서 count 컬럼을 제외한다.
train_csv = train_csv.dropna() # 행에 결측치가 하나라도 있을땐 행전체를 삭제한다.
verbose = 0 # 훈련화면을 모두 출력하지 않아 훈련속도가 가장 빠름 1= 디폴트값, 2=프로그레스바 삭제, 그 외 숫자= 에포 도는 화면만 보여줌
validation data = 검증을 위한 데이터. train data의 학습검증용. 성능이 올라가고 통상적으로 train data의 30%정도까지 사용.
model.fit(x_train, y_train, epochs = 00, batch_size= 32, validation_data = (x-val, y_val)) #의 형태로 사용.
print("음수 갯수 :", submission_csv[submission_csv['count']<0].count()) # submission_csv 에서 0보다 작은 count 컬럼의 갯수를 보여달라는 뜻.

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  #mse(y_test, y_predict)에 루트를 씌운다.
rmse = RMSE(y_test, y_predict)
print("RMSE", rmse)

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict)) #msle(y_test, y_predict)에 루트를 씌운다.
rmsle = RMSLE (y_test, y_predict)
print("RMSLE", rmsle)


