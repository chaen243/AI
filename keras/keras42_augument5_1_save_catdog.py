
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
import tensorflow as tf



start_time = time.time()
#1 데이터

path = 'C:\\_data\\image\\CatDog\\'
train_path = path + "train\\"
test_path = path + "Test\\"


BATCH_SIZE = int(1000)
IMAGE_SIZE = int(150)

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)

xy_train_data = train_data_gen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
)

test_data_gen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_data_gen.flow_from_directory(
    test_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=99999,
    class_mode='binary',
    shuffle=False
)


x, y = np.vstack([xy_train_data])

split_num = np.random.randint(x.shape[0],size=10000)

splited_x = x[split_num].copy()
splited_y = y[split_num].copy()

aug_data_gen = ImageDataGenerator(
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    shear_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
)

aug_data = aug_data_gen.flow(
    splited_x,
    splited_y,
    batch_size=x.shape[0],
    shuffle=False
)

aug_x, aug_y = aug_data.next()

print(f"{aug_x.shape=} {aug_y.shape=}")

x = np.concatenate([x,aug_x])
y = np.concatenate([y,aug_y])

print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_aug_x.npy",arr=x)
np.save(save_path+"_aug_y.npy",arr=y)
np.save(save_path+"_test.npy",arr=test_data[0][0])



print('finish! :')
'''

#2 모델구성
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape = (200,200,3) , strides=2 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2), activation='relu' , strides=2  ))
model.add(MaxPooling2D())
model.add(Conv2D(28,(2,2), activation='relu', strides=2  ))
model.add(MaxPooling2D())
model.add(Conv2D(24,(2,2), activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(20,(1,1), activation='relu' ))
model.add(Flatten())
model.add(Dense(80,activation='relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
filepath = "C:\_data\_save\MCP\_k39\CatDog"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

es = EarlyStopping(monitor='val_acc' , mode = 'auto' , patience= 10 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_acc', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 100 , batch_size= 5 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])

path = 'C:\\_data\\_save\\MCP\\_k39\\CatDog\\'


#4 평가, 예측
import os


loss = model.evaluate(x_test,y_test)
y_prediect = model.predict(test)
y_prediect = np.around(y_prediect.reshape(-1))
print(y_prediect.shape)




id_list = os.listdir(test_path)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)

y_submit = pd.DataFrame({'id':id_list,'Target':y_prediect})
#print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{loss[1]:.6f}.csv",index=False)



end_time = time.time()
print('걸린시간 : ' , round(end_time - start_time,2), "초" )
print('loss:',loss)



y_submit = pd.DataFrame({'id': id_list,'Target':y_prediect})
#print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{loss[1]:.6f}.csv",index=False)








import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['val_acc'], c = 'pink', label = 'val_acc', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')

plt.title('cat_dog')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.grid
plt.show()


# loss [0.5740843415260315, 0.7079663872718811]
# ACC :  0.7079663730984788
# 걸린시간 :  241.63 초
# 변환시간 :  111.52 초


#증폭
'''