
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



start_time = time.time()
#1 데이터
np_path = "C:\\_data\\_save_npy\\"

x_train = np.load(np_path + 'keras39_3_x_train.npy')
y_train = np.load(np_path + 'keras39_3_y_train.npy')
test = 'C:\_data\image\CatDog\Test'



x_train , x_test, y_train , y_test = train_test_split(
    x_train, y_train,random_state= 3702 , shuffle= True,
    stratify=y_train)


# train_datagen = ImageDataGenerator(rescale=1./255,)

#                                 #    horizontal_flip= True ,
#                                 #    vertical_flip = True ,
#                                 #    width_shift_range = 0.1,
#                                 #    height_shift_range = 0.1,
#                                 #    rotation_range = 5 ,
#                                 #    zoom_range = 1.2,
#                                 #    shear_range = 0.8 ,
#                                 #    fill_mode = 'nearest')


# test_datagen = ImageDataGenerator(rescale=1./255) #데이터 수치화 도구! #수치가 같아야 훈련이 가능하기에 testdata도 똑같이 수치화는 해줘야함.


# start_time2 = time.time()
# xy_train = train_datagen.flow_from_directory(path_train, 
#                                              target_size = (500,500), #원본데이터보다 작아질수록 성능이많이 떨어짐. 최대한 원본과 사이즈를 맞춰주는게 좋음.
#                                              batch_size = BATCH_SIZE,
#                                              class_mode='binary', 
#                                              shuffle=True)


# xy_test = test_datagen.flow_from_directory(
#     path_test, target_size=(400, 400),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     shuffle=False,)

# x = np.array(np.arange(0,27)).reshape(3,3,3,1)
# y = np.array(np.arange(0,27)).reshape(3,3,3,1)

# xy = np.vstack([x,y])

# x_train = xy_train[0][0]  
# y_train = xy_train[0][1]
# # x_test = xy_test[0][0]
# # y_test = xy_test[0][1]

# x=[]
# y=[]
# failed_i = []

# for i in range(int(20000 / BATCH_SIZE)):
#     try: 
#         xy_data = xy_train.next()
#         new_x = xy_data[0]
#         new_y = xy_data[1]
#         if i==0:
#             x = np.array(new_x)
#             y = np.array(new_y)
#             continue
        
#         x = np.vstack([x,new_x])
#         y = np.hstack([y,new_y])
#         print("i: ",i)
#         print(f"{x.shape=}\n{y.shape=}")
#     except:
#         print("failed i: ",i)
#         failed_i.append(i)
        

# x_train , x_test, y_train , y_test = train_test_split(
#     xy_train[0][0], xy_train[0][1] ,random_state= 3702 , shuffle= True,
#     stratify=xy_train[0][1])



end_time2 = time.time()









print(x_train.shape)




#2 모델구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape = (500,500,3) , strides=1 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2), activation='relu' ))
model.add(Conv2D(28,(2,2), activation='relu' ))
model.add(Conv2D(24,(2,2), activation='relu' ))
model.add(Conv2D(20,(2,2), activation='relu' ))
model.add(BatchNormalization())
model.add(Conv2D(16,(2,2), activation='relu' ))
model.add(Conv2D(12,(2,2), activation= 'relu' ))
model.add(MaxPooling2D())
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

es = EarlyStopping(monitor='val_loss' , mode = 'auto' , patience= 100 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 1000 , batch_size= 50 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])


#4 평가, 예측
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss',result)
end_time = time.time()
print('걸린시간 : ' , round(end_time - start_time,2), "초" )


import os
path = 'C:\\_data\\image\\CatDog\\'

forder_dir = path+"test\\test"
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)

y_submit = pd.DataFrame({'id':id_list,'Target':y_predict})
print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{result[1]:.6f}.csv",index=False)







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
