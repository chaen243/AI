import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]).reshape(-1,3,1)

x = x.reshape(-1,3,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= True, random_state = 442)


#2. 모델구성
model = Sequential()
model.add(LSTM(256, return_sequences=True,
               input_shape= (3,1), activation = 'relu'))
#return_sequences로 2개이상의 LSTM을 사용 할 수 있다.
model.add(LSTM(8, return_sequences=True))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1))

model.summary()
#3. 컴파일, 훈련
es = EarlyStopping(monitor= 'loss', mode= 'auto', patience = 200, verbose= 0, restore_best_weights= True )
model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc')
model.fit(x,y,epochs=2500, callbacks= [es], batch_size= 1)

#4. 평가, 훈련
results = model.evaluate(x,y)
print ('loss:', results)
y_pred = np.array([50,60,70]).reshape(1,3,1)
y_pred = model.predict(y_pred)
print('예측값 :', y_pred)


# loss: [0.008794952183961868, 0.0]
# 예측값 : [[80.0718]]

# loss: [0.0038003113586455584, 0.0]
# 예측값 : [[80.91556]]

# loss: [0.002848625648766756, 0.0]
# 예측값 : [[80.96809]]



# loss: [5.7165038015227765e-05, 0.0]
# 예측값 : [[73.74017]]