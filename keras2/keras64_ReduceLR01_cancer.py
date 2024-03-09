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

# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 그래서 삭제
#pip uninstall scikit-learn
#pip uninstall scikit-image                  
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==1.1.3


#pip install scikit-learn==0.23.2 사이킷런 0.23.2버전을 새로 깐다.

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
import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path= '../_data/_save/MCP/_k25/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path, 'k25_', date, "_", filename]) #""공간에 ([])를 합쳐라.


from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


model.compile(loss= 'binary_crossentropy', optimizer= Adam(learning_rate = 0.1) ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82

es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'auto', 
                   patience = 20, 
                   verbose = 1, 
                   restore_best_weights= True)

mcp = ModelCheckpoint(monitor='val_loss', 
                      mode = 'auto', verbose= 1, 
                      save_best_only= True, 
                      filepath= filepath,
                      period = 20)


##########################################
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience= 10,
                        mode= 'auto',
                        factor= 0.5,
                        verbose=1)
#es의 patience보다 적게 잡을것.
##########################################

hist = model.fit(x_train, y_train, callbacks=[es,mcp,rlr], epochs= 1000, batch_size = 20, validation_split= 0.27)


#4. 평가, 예측
print("=============기본출력===============")
results = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0) 
r2 = r2_score(y_test, y_predict)
print("로스 :", results)
print("R2 스코어 :", r2)


#print("걸린 시간 :", round(end_time - start_time, 2), "초") #def로 정의하지 않은 함수는 파이썬에서 기본으로 제공해주는 함수.




print('===================================')
print('val_loss:', hist.history['val_loss'][-20])
print('===================================')

