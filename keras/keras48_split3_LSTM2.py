import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

a = np.array(range(1, 101))
x_pred = np.array(range(96, 106)) #96~105
size = 5  #x데이터4개 y데이터 1개



def split_x(dataset, size): #a, timesteps만큼 자름
    aaa = []                #aaa = 
    for i in range(len(dataset) - size +1): # i = data길이 - timesteps +1
        subset = dataset[i : (i + size)]
        aaa.append(subset)                  # 반복
        
    return np.array(aaa)



xy_splited = split_x(a, size)
x = xy_splited[:, :-1].reshape(-1,2,2)
y = xy_splited[:, -1]

x_predict = split_x(x_pred,size-1).reshape(-1,2,2)
print(x_predict.shape)


print(x.shape) #(96, 2, 2)

print(x)


#2. 모델구성
model = Sequential()
model.add(LSTM(256, return_sequences=False,
               input_shape= (2,2), activation = 'relu'))
#return_sequences로 2개이상의 LSTM을 사용 할 수 있다.
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1))

model.summary()
#3. 컴파일, 훈련
es = EarlyStopping(monitor= 'loss', mode= 'auto', patience = 200, verbose= 0, restore_best_weights= True )
model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc')
model.fit(x,y,epochs=20, callbacks= [es], batch_size= 1)

#4. 평가, 훈련
results = model.evaluate(x)
print ('loss:', results)
x_pred = model.predict(x_predict)
print('예측값 :', x_pred)


# 예측값 : [[ 98.95157 ]
#  [ 99.94138 ]
#  [100.93533 ]
#  [101.92128 ]
#  [102.919014]
#  [103.91304 ]
#  [104.89963 ]]

    

    