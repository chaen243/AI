# x, y 추출해서 모델만들기
# 성능 0.99 이상

#데이터 증폭의 한종류.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
rescale=1/255.,
horizontal_flip=True,    #수평 뒤집기
vertical_flip=True,      #수직뒤집기
width_shift_range=0.1,   #가로로 0.1만큼 평행이동
height_shift_range=0.1,  #세로로 0.1만큼평행이동  
rotation_range=5,        #정해진각도만큼 이미지를 회전
zoom_range=1.2,          #축소 또는 확대
shear_range=0.7,         #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
fill_mode='nearest',     #빈공간을 근사값(가장 비슷한 색)으로 채워준다
)

test_datagen = ImageDataGenerator( #테스트 데이터는 실제 값이어야 하기때문에 rescale만 하고끗!
    rescale=1/255.
)

path_train = 'C:\\_data\\image\\brain\\train\\' #path 하단 폴더는 라벨(y값)로 지정됨.
path_test = 'C:\\_data\\image\\brain\\test\\' 

xy_train = train_datagen.flow_from_directory( #폴더 안에 있는 칭구들 수치화 해줌.
    path_train, target_size=(200, 200), #모든 타겟의 사이즈를 target size의 size로 맞춰준다.
    batch_size=160, #크게 잡으면(현 데이터에서는 160이상) 통데이터만큼 가져와줌.
    class_mode='binary',
    shuffle=True,
)# Found 160 images belonging to 2 classes. 수치화된 160개의 이미지에 (160, )의 y가 생성됨/
 
# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001CE07AB3520> 래핑되어있는 형태여서 현재상태로는 데이터를 볼 수 없음.
# x,y가 합쳐져있는 데이터 형태임.

xy_test = test_datagen.flow_from_directory( #폴더 안에 있는 칭구들 수치화 해줌.
    path_test, target_size=(200, 200), #훈련데이터와 사이즈를 맞춘것. 데이터조작x
    batch_size=160,
    class_mode='binary',
    shuffle=False,
)#Found 120 images belonging to 2 classes.

#print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x00000198923DD490>

#print(xy_train.next()) #두개 똑같음.
#print(xy_train[16]) #^^^ 배치사이즈=10으로 잘림. 전체데이터160/10 으로 16개. [16]은17번째 값이어서 에러가 나옴.
#Asked to retrieve element 17, but the Sequence has length 16

#print(xy_train[0][0]) #첫번째 배치의 x값
#print(xy_train[0][1]) #첫번째 배치의 y값
#print(xy_train[0][0].shape) #(10, 200, 200, 3) 흑백== 칼라로 인식 가능. 칼라는 흑백 인식 불가.

#print(type(xy_train)) #Iterater형태
#print(type(xy_train[0])) #<class 'tuple'>
#print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
#print(type(xy_train[0][1])) #<class 'numpy.ndarray'>



x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]


#print(x_train.shape, y_train.shape) #(160, 200, 200, 3) (160,)
#print(x_test.shape, y_test.shape) #(120, 200, 200, 3) (160,)



#2. 모델구성
model = Sequential()
model.add(BatchNormalization())
model.add(Conv2D(24, (2,2), activation='sigmoid', input_shape=(200, 200, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu',strides=2))
model.add(MaxPooling2D())
model.add(Conv2D(12, (3,3), activation='relu',padding='same',strides=2))
model.add(MaxPooling2D())
# Stack 2
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

filepath = "C:\_data\_save\MCP\_k37"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

#x_train = x_train.reshape ( (x_train.shape[0], 32, 32, 3))
#x_test = x_test.reshape ( (x_test.shape[0], 32, 32, 3))

#3. 컴파일, 훈련


model.compile( loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['acc'])
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 200, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit(x_train, y_train, batch_size=100, verbose=2, validation_data= (x_test,y_test), epochs= 2000,shuffle=True, callbacks= [es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )


# loss: 0.10842730104923248
# acc: 0.9416666626930237