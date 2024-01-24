#https://www.kaggle.com/playlist/men-women-classification

#데이터 경로
# _data/kaggle/men_women/
# 남자여자 확인해서 결과치 메일로 보낼것. 

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time

start_time = time.time()


#1. 데이터


np_path = "C:\\_data\\_save_npy\\manwoman\\"

x_train = np.load(np_path + 'keras39_5_x_train.npy')
y_train = np.load(np_path + 'keras39_5_y_train.npy')
x_test = np.load(np_path + 'keras39_5_x_test.npy')
y_test = np.load(np_path + 'keras39_5_y_test.npy') 





#2 모델구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape = (250,250,3) , strides=1 , activation='relu' ))
model.add(Dropout(0.1))
model.add(Conv2D(32,(2,2), activation='relu' , padding = 'same'))
model.add(Conv2D(12,(2,2), activation='relu' ))
model.add(Conv2D(12,(3,3), activation= 'relu' ))
model.add(Flatten())
model.add(Dense(80,activation='relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
filepath = "C:\\_data\\_save\MCP\\_k39\\man_woman\\"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

es = EarlyStopping(monitor='val_loss' , mode = 'auto' , patience= 100 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs = 1000 , batch_size= 10 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])


#4 평가, 예측
result = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

print('loss',result[0])
print('loss',result[1])

y_test = np.round(y_test)
y_predict = np.round(y_predict)

print('ACC : ' , accuracy_score(y_test,y_predict))

end_time = time.time()

print('걸린시간 : ' , round(end_time - start_time,2), "초" )



# loss [0.5740843415260315, 0.7079663872718811]
# ACC :  0.7079663730984788
# 걸린시간 :  241.63 초
# 변환시간 :  111.52 초
