
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
path1 = 'C:\\_data\\image\\CatDog\\'
train_path = path1 + "train\\"
test_path = path1 + "test\\"



np_path = "C:\\_data\\_save_npy\\"

x = np.load(np_path + 'data_200px_x.npy')
y = np.load(np_path + 'data_200px_y.npy')
test = np.load(np_path+ "data_200px_test.npy")


x_train , x_test, y_train , y_test = train_test_split(
    x, y, train_size=0.7, random_state= 3702 , shuffle= True,
    stratify=y)





end_time2 = time.time()









print(x_train.shape)




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

es = EarlyStopping(monitor='val_acc' , mode = 'auto' , patience= 100 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_acc', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 15 , batch_size= 5 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])

path = 'C:\\_data\\_save\\MCP\\_k39\\CatDog\\'


#4 평가, 예측
import os


loss = model.evaluate(x_test,y_test,verbose=0)
y_prediect = model.predict(test)
y_prediect = np.around(y_prediect.reshape(-1))
print(y_prediect.shape)

print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")

model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")



forder_dir = path1+"test\\"
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)

y_submit = pd.DataFrame({'id':id_list,'Target':y_prediect})
print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{loss[1]:.6f}.csv",index=False)



end_time = time.time()
print('걸린시간 : ' , round(end_time - start_time,2), "초" )



path = 'C:\\_data\\image\\CatDog\\'

forder_dir = path1+"test\\test"
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)
    


y_submit = pd.DataFrame({'id': id_list,'Target':y_prediect})
print(y_submit)
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
