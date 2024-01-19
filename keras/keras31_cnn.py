from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D


#컬러사진 10x10사이즈 10000장이 있다고 한다면 shape는 (10000,10,10,3)


model=Sequential()
#model.add(Dense(10, input_shape=(3, ))) # 인풋= (n,3)
model.add(Conv2D(10, (2,2), input_shape=(10,10,1)))  #10=아웃풋값/ (2,2)는 조각내주는 사이즈(가중치의 shape/curnelsize)/ input_shape= 가로10x세로10사이즈의 흑백사진.(마지막에 3이면 컬러사진)
model.add(Dense(5))
model.add(Dense(1))
