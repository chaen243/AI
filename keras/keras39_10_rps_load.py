import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import to_categorical

#1. 데이터

np_path = "C:\\_data\\_save_npy\\"

x = np.load( np_path + 'rps_x.npy')
y = np.load( np_path + 'rps_y.npy')

#print(x.shape) #(2520, 150, 150, 3)
#print(y.shape) #(2520, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state=234, shuffle=True, stratify= y)

#print(x_train.shape) #(2016, 150, 150, 3)


#2. 모델구성
model = Sequential()
model.add(Conv2D(12,(2,2),input_shape = (150,150,3) , strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(12,(2,2), activation='relu' , ))
model.add(MaxPooling2D())
model.add(Conv2D(24,(2,2), activation='relu', ))
model.add(MaxPooling2D())
model.add(Conv2D(128,(2,2), activation='relu',  ))
model.add(Conv2D(128,(2,2), activation='relu', ))
model.add(Conv2D(128,(2,2), activation='relu', ))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(24,activation='relu'))
model.add(Dense(3,activation='softmax'))

#model.summary()


#3 컴파일, 훈련
filepath = "c:\\_data\\_save\\MCP\\_k39\\rps"

es = EarlyStopping(monitor='val_loss', mode= 'auto', patience= 30, restore_best_weights= True, verbose= 1)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only= True, filepath= filepath )

model.compile(loss= 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 100 , batch_size= 128 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])



#4. 평가, 예측

result = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict)

print("로스 :", result[0])
print("정확도 :", result[1])


#로스 : 3.96894829464145e-05
#정확도 : 1.0