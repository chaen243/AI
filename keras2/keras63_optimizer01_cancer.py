'''
learning_rate는 초반에는 크게 잡을것.
learning_rate가 작으면 local minimum에 빠지기 쉬움.
너무 작으면 local minimum까지 가지 못할수도 있다.

learning rate가 크면 속도가 빨라지고 정확도가 올라감.
너무 크게 되면 미니멈 위쪽에서 핑퐁치고 있을 확률이 높음.

ReduceLR - 처음엔 LR을 크게 주고 나중에는 감소시키겠다.
'''
from sklearn.datasets import load_boston, load_breast_cancer

from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time

datasets = load_breast_cancer()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)

print(datasets.DESCR) #Describe #행 = Tnstances




#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20,shuffle=True, random_state= 4041, stratify= y ) #4041




print(x_train)          
print(y_train)
print(x_test)
print(x_test)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)




#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim= 30))
model.add(Dense(9))
model.add(Dense(13))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1, activation= 'sigmoid'))

model.summary()





# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
learning_rate = 1.0 
# learning_rate = 0.1 
#learning_rate = 0.01 
#learning_rate = 0.001 
#learning_rate = 0.0001 

model.compile(loss= 'binary_crossentropy', optimizer= Adam(learning_rate=learning_rate) ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82

##########################################
model.fit(x_train, y_train, epochs= 200, batch_size = 20, validation_split= 0.27)


#4. 평가, 예측
print("=============기본출력===============")
loss = model.evaluate(x_test, y_test, verbose=0)
print("lr : {0}, 로스 :{1}".format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0) 
acc = accuracy_score(y_test, np.around(y_predict))
print("lr : {0}, acc :{1}".format(learning_rate, acc))

#100
# lr : 1.0, 로스 :805321792.0
# lr : 1.0, acc :0.956140350877193

# lr : 0.1, 로스 :19590.67578125
# lr : 0.1, acc :0.9649122807017544

# lr : 1.0, 로스 :13097439232.0
# lr : 1.0, acc :0.956140350877193

# lr : 0.1, 로스 :8656.9501953125
# lr : 0.1, acc :0.9473684210526315

# lr : 0.01, 로스 :0.12923549115657806
# lr : 0.01, acc :0.9824561403508771

# lr : 0.001, 로스 :0.1382901668548584
# lr : 0.001, acc :0.9649122807017544

# lr : 0.0001, 로스 :0.15554079413414001
# lr : 0.0001, acc :0.956140350877193

#200
# lr : 1.0, 로스 :805321792.0
# lr : 1.0, acc :0.956140350877193

# lr : 0.01, 로스 :1.5502671003341675
# lr : 0.01, acc :0.956140350877193

# lr : 0.001, 로스 :1.2891976833343506
# lr : 0.001, acc :0.9298245614035088

# lr : 0.0001, 로스 :0.15464836359024048
# lr : 0.0001, acc :0.9298245614035088