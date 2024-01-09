import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col= 0) 
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
submission_csv = pd.read_csv(path + "submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, shuffle= False, random_state= 334)


#2. 모델구성
model = Sequential()
model.add(Dense(1024, input_dim = 9, activation= 'relu'))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64,))
model.add(Dense(32,))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(4, activation= 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile (loss = 'mse', optimizer= 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs= 100, batch_size = 10, validation_split= 0.2)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)

print("로스 :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
#submission_csv['count'] = y_submit
#print(submission_csv)

#submission_csv.to_csv(path + "submission__45.csv", index= False)

#import time as tm
#ltm = tm.localtime(tm.time())
#save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
#submission_csv.to_csv(path + f"submission_{save_time}.csv", index=False)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize= (9,6))
plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.title("따릉이 LOSS")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()





