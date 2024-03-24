
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time



start_time = time.time()
#1 데이터


train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip= True ,
                                   vertical_flip = True ,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 5 ,
                                   zoom_range = 1.2,
                                   shear_range = 0.8 ,
                                   fill_mode = 'nearest')


test_datagen = ImageDataGenerator(rescale=1./255) #데이터 수치화 도구! #수치가 같아야 훈련이 가능하기에 testdata도 똑같이 수치화는 해줘야함.

path_train = 'c://_data//image//catdog//Train//'
path_test = 'c://_data//image//catdog//Test//'

start_time2 = time.time()
xy_train = train_datagen.flow_from_directory(path_train, 
                                             target_size = (180,180), #원본데이터보다 작아질수록 성능이많이 떨어짐. 최대한 원본과 사이즈를 맞춰주는게 좋음.
                                             batch_size = 20000,
                                             class_mode='binary', 
                                             shuffle=True)


xy_test = test_datagen.flow_from_directory(
    path_test, target_size=(180, 180),
    batch_size=20000,
    class_mode='binary',
    shuffle=False,)



x_train = xy_train[0][0]  
y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]


x_train , x_test, y_train , y_test = train_test_split(
    xy_train[0][0], xy_train[0][1] ,random_state= 3702 , shuffle= True,
    stratify=xy_train[0][1])



end_time2 = time.time()


#2 모델구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape = (180,180,3) , strides=1 , activation='relu' ))
model.add(Conv2D(128,(2,2), activation='relu' ))
model.add(Conv2D(64,(2,2), activation='relu' ))
model.add(Conv2D(12,(3,3), activation= 'relu' ))
model.add(Flatten())
model.add(Dense(80,activation='relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
filepath = "C:\_data\_save\MCP\_k37"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 10 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs = 30 , batch_size= 500 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])


#4 평가, 예측
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss',result)

y_test = np.round(y_test)
y_predict = np.round(y_predict)

print('ACC : ' , accuracy_score(y_test,y_predict))

end_time = time.time()

print('걸린시간 : ' , round(end_time - start_time,2), "초" )
print('변환시간 : ' , round(end_time2 - start_time2,2), "초" )

# #gpu
# loss [0.6873248815536499, 0.5471367835998535]
# ACC :  0.547136784196049
#걸린시간 :  166.35 초