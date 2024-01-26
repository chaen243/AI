from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical


#acc- 0.77이상

#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    #rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest', #nearest가 디폴트 #reflet 좌우반전,wrap 감싸줌, 
)

#print(x_train.shape) #(50000, 32, 32, 3)
#print(x_test.shape) #(10000, 32, 32, 3)

augumet_size = 50000

randidx = np.random.randint(x_train.shape[0],size=augumet_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
print(x_augumented.shape) #(50000, 32, 32, 3)
print(y_augumented.shape) #(50000, 1)


x_augumented = x_augumented.reshape(-1, 32, 32, 3)



x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    save_to_dir= 'c:\\_data\\temp\\',
    shuffle= False #섞이면 데이터가틀어짐.
).next()[0]

#print(x_augumented)
print(x_augumented.shape)

#print(x_train.shape)
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)

x_train = np.concatenate((x_train,x_augumented))
y_train = np.concatenate((y_train,y_augumented))



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer

#mms = MinMaxScaler(feature_range=(0,1))
#mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


#mms.fit(x_train)
#mms.fit(x_test)
#x_train= mms.transform(x_train)
#x_test= mms.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델

model = Sequential()
# model.add(Conv2D(32, (3,3),
#                  input_shape= (32, 32, 3), activation= 'relu')) #첫 아웃풋 = filter
# # shape = (batch_size, rows, columns, channels)
# # shape = (batch_size, heights, widths, channels)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3,3), activation= 'relu')) 
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (2,2), activation= 'relu')) 
# model.add(Conv2D(32, (2,2), activation= 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2))) 
# model.add(Flatten()) #(n,22*22*15)의 모양을 펴져있는 모양으로 변형.
# # shape = (batch_size(=model.fit의 batch_size와 같음.), input_dim) 
# model.add(Dense(10, activation= 'softmax'))
# Convolutional Block (Conv-Conv-Pool-Dropout)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), strides= 2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block (Conv-Conv-Pool-Dropout)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'), )
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Classifying
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.summary()


filepath = "C:\_data\_save\MCP\_k31"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

x_train = x_train.reshape ( (-1, 32, 32, 3))
x_test = x_test.reshape ( (-1, 32, 32, 3))

#3. 컴파일, 훈련
model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 299, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit( x_train, y_train, batch_size=6000, verbose=2, epochs= 500, validation_split=0.3, callbacks= [es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )

# basic
# loss: 0.6674924492835999
# acc: 0.7788000106811523
# 걸린시간 : 459.2906882762909 초

# # #stride2,padding same
# loss: 0.9534503817558289
# acc: 0.6705999970436096
# 걸린시간 : 167.60450172424316 초

#MaxPooling
# loss: 0.7770913243293762
# acc: 0.7664999961853027
# 걸린시간 : 169.09292793273926 초

#증폭
#loss: 0.9374784231185913
#acc: 0.6732000112533569
