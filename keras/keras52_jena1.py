import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import time


#1. 데이터
path = "C:/_data/kaggle/jena/"

dataset = pd.read_csv(path + 'jena.csv', index_col= 0)

#print(dataset.columns)
# 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',      
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',   
#       'wd (deg)'],


#print(type(dataset)) #<class 'pandas.core.frame.DataFrame'>

timestep = 720

#5일분(720) 훈련시켜 하루 뒤(144) 를 예측


def split_xy(dataset, timestep, y_column):
    x, y = list(), list()
    
    for i in range(len(dataset)-timestep):
        x.append(dataset[i : i+ timestep])
        y.append(dataset.iloc[i+ timestep][y_column])
        
    return np.array(x), np.array(y)
x, y = split_xy(dataset, timestep, 'T (degC)')    

print(x.shape) #(420548, 3, 14)
print(y.shape) #(420548,)





# timestep = 720
# predict_step = 144



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle= False)
print(x_train.shape) #(294383, 3, 14)
print(x_test.shape) #(126165, 3, 14)

mms = MinMaxScaler()
mms2 = MinMaxScaler()


mms2.fit(y.reshape(-1,1))
dataset= mms.fit_transform(dataset)






#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=x_train.shape[1:]))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1))
#model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss' , mode = 'auto' , patience= 100 , restore_best_weights=True , verbose= 1  )


start_time = time.time()

model.compile(loss= 'mse' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs = 10 , batch_size= 800 , validation_split= 0.2, verbose= 2 ,callbacks=[es])

model.save("c:\_data\_save\keras52_jena_save_model.h5")

end_time = time.time()

#4 평가, 예측
result = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print(y_predict)
print('loss :',result[0])
print('r2 :', r2)

# print('걸린시간 : ' , round(end_time - start_time,2), "초" )



#y값도 scaler가 되서 inverse_transform 해주기
predicted_degC = mms2.inverse_transform(np.array(y_predict).reshape(-1,1))
y_true = mms2.inverse_transform(np.array(y_test).reshape(-1,1))
print(x_test.shape,y_predict.shape) #(126165, 3, 14) (126165, 1)
print(predicted_degC.shape) #(126165, 1)



# GRU
# loss : 0.1745879352092743
# r2 : 0.997130423880108
# 걸린시간 :  6.93 초

# SimpleRNN
# loss : 0.13116677105426788
# r2 : 0.997844108257324
# 걸린시간 :  7.64 초

# Bidirectional
# loss : 0.2529450058937073
# r2 : 0.9958425248480712
# 걸린시간 :  10.32 초

# LSTM
# loss : 0.15823504328727722
# r2 : 0.9973992038646582
# 걸린시간 :  7.47 초

