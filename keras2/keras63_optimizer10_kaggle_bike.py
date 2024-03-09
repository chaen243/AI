import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(submission_csv.shape) # (6493, 2)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())


x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.77, shuffle = False, random_state=1266)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620, ) (3266, )


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1024, input_dim = 8))
model.add(Dropout(0.3))
model.add(Dense(512,))
model.add(Dense(256,))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,  activation='relu'))

#3. 컴파일, 훈련
import datetime
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 400, verbose = 0, restore_best_weights= True)

model.compile(loss= 'mse', optimizer= Adam(), metrics='mse' ) 
hist = model.fit(x_train, y_train, callbacks=[es], epochs= 10000, batch_size = 100, validation_split= 0.27)
end_time = time.time()





#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)

loss = model.evaluate(x_test, y_test, verbose=0)
print("lr : {0}, 로스 :{1}".format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0) 
acc = accuracy_score(y_test, np.around(y_predict))
print("lr : {0}, acc :{1}".format(learning_rate, acc))



#print(y_submit)
#print(y_submit.shape) #(6493, 1)
#print(submission_csv.shape) #(6493, 2)




print("걸린 시간:", round(end_time - start_time, 2), "초")

submission_csv['count'] = y_submit

print(submission_csv)
accuracy_score = ((y_test, y_submit))

y_submit = (y_submit.round(0).astype(int))


#submission_csv.to_csv(path + "submission_29.csv", index= False)
print("음수갯수 :", submission_csv[submission_csv['count']<0].count())
print("로스 :", loss)
print("R2 스코어 :", r2)
print("정확도 :",accuracy_score)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict)
print("로스 :", loss)
print("RMSLE :", rmsle)

path = "C:\\_data\\kaggle\\bike\\"

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmsle:.3f}.csv", index=False)
#MSE : 23175.111328125
#R2 스코어 : 0.27044473122031987
#RMSE :  152.23374956711748
#RMSLE : 1.3152084898668681

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize= (9,6))
plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.title("케글바이크 LOSS")
plt.xlabel('epoch')
plt.grid()
plt.show()


