import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time
from keras.utils import to_categorical

#1. 데이터

np_path = "C:\\_data\\_save_npy\\"

x = np.load(np_path + 'horse_human_x.npy')
y = np.load(np_path + 'horse_human_y.npy')

print(x.shape) #(1027, 300, 300, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 234, shuffle= True, stratify= y)

print(x_train.shape)
print(y_train.shape)



#2. 모델구성
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape = (300,300,3) , strides=2 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(24,(2,2), activation='relu' ,  ))
model.add(MaxPooling2D())
model.add(Conv2D(20,(2,2), activation='relu',  ))
model.add(MaxPooling2D())
model.add(Conv2D(24,(2,2), activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(20,(1,1), activation='relu' ))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2,activation='softmax'))

#model.summary()

filepath = "C:\_data\_save\MCP\_k39\horse_human"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

es = EarlyStopping(monitor='val_acc' , mode = 'auto' , patience= 100 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_acc', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 250 , batch_size= 16 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])




#4 평가, 예측
import os


result = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict)

print("로스 :", result[0])
print("정확도 :", result[1])

#로스 : 0.006262290291488171
#정확도 : 1.0


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize= (9,6))
# plt.plot(hist.history['f1_score'], c = 'red', label = 'f1', marker = '.')
plt.plot(hist.history['val_acc'], c = 'pink', label = 'val_acc', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')

plt.legend(loc = 'upper right')
plt.title("hh LOSS")
plt.xlabel('epoch')
plt.grid()
plt.show()

#로스 : 0.054071489721536636
#정확도 : 0.9951456189155579