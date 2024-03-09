from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization,Conv2D,Flatten, MaxPooling2D, LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import sklearn as sk
import time
import warnings
warnings.filterwarnings(action='ignore')
from keras.utils import to_categorical




#1. 데이터
le = LabelEncoder()
path = "C:\\_data\\daicon\\bank\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
#print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
#print(test_csv)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
#print(submission_csv)


# print(train_csv.shape) #(96294, 14)
# print(test_csv.shape)  #(64197, 13)
# print(submission_csv.shape) #(64197, 2)


#print(train_csv.columns) #'대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#       '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'

#대출기간 처리
train_loan_time = train_csv['대출기간']
train_loan_time = train_loan_time.str.split()
for i in range(len(train_loan_time)):
    train_loan_time.iloc[i] = int((train_loan_time)[i][0])
    
#print(train_loan_time)   

test_loan_time = test_csv['대출기간']
test_loan_time = test_loan_time.str.split()
for i in range(len(test_loan_time)):
    test_loan_time.iloc[i] = int((test_loan_time)[i][0])

#print(test_loan_time)   

train_csv['대출기간'] = train_loan_time
test_csv['대출기간'] = test_loan_time



#print(test_csv)


#근로기간 처리
train_working_time = train_csv['근로기간']
test_working_time = test_csv['근로기간']

for i in range(len(train_working_time)):
    data = train_working_time.iloc[i]
    if data == 'Unknown':
        train_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_working_time.iloc[i] = int(15)
    elif data == '< 1 year' or data == '<1 year':
        train_working_time.iloc[i] = int(0.6)
    else:
        train_working_time.iloc[i] = int(data.split()[0])
        
    
train_working_time = train_working_time.fillna(train_working_time.mean())

#print(train_working_time)

for i in range(len(test_working_time)):
    data = test_working_time.iloc[i]
    if data == 'Unknown':
        test_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_working_time.iloc[i] = int(15)
    elif data == '< 1 year' or data == '<1 year':
        test_working_time.iloc[i] = int(0.6)
    else:
        test_working_time.iloc[i] = int(data.split()[0])
    
test_working_time = test_working_time.fillna(test_working_time.mean())

train_csv['근로기간'] = train_working_time
test_csv['근로기간'] = test_working_time 
#print(test_csv['근로기간'])
#print(train_csv['근로기간'])




#대출목적 전처리
test_loan_perpose = test_csv['대출목적']
train_loan_perpose = train_csv['대출목적']



for i in range(len(test_loan_perpose)):
    data = test_loan_perpose.iloc[i]
    if data == '결혼':
        test_loan_perpose.iloc[i] = np.NaN
        

for i in range(len(test_loan_perpose)):
    data = test_loan_perpose.iloc[i]
    if data == '기타':
        test_loan_perpose.iloc[i] = np.NaN
               
for i in range(len(train_loan_perpose)):
    data = train_loan_perpose.iloc[i]
    if data == '기타':
        train_loan_perpose.iloc[i] = np.NaN
        

test_loan_perpose = test_loan_perpose.fillna(method='bfill')
train_loan_perpose = train_loan_perpose.fillna(method='ffill')


test_csv['대출목적'] = test_loan_perpose
train_csv['대출목적'] = train_loan_perpose







######## 결측치확인
# print(test_csv.isnull().sum()) #없음.
# print(train_csv.isnull().sum()) #없음.





#print(np.unique(test_loan_perpose))  #'기타' '부채 통합' '소규모 사업' '신용 카드' '의료' '이사' '자동차' '재생 에너 
#지' '주요 구매' '주택'
# '주택 개선' '휴가'







le.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le.transform(train_csv['주택소유상태'])
le.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = le.transform(test_csv['주택소유상태'])
#print(train_csv['주택소유상태'])
#print(test_csv['주택소유상태'])


le.fit(train_csv['대출목적'])
train_csv['대출목적'] = le.transform(train_csv['대출목적'])
le.fit(test_csv['대출목적'])
test_csv['대출목적'] = le.transform(test_csv['대출목적'])

le.fit(train_csv['대출기간'])
train_csv['대출기간'] = le.transform(train_csv['대출기간'])
le.fit(test_csv['대출기간'])
test_csv['대출기간'] = le.transform(test_csv['대출기간'])





x = train_csv.drop(['대출등급'], axis = 1)


#print(x)
y = train_csv['대출등급']


y = le.fit_transform(y)
     




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.83,  shuffle= True, random_state= 279, stratify= y) #170 #279 


  

y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7) 
#민맥스 - 스탠다드
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer


mms = RobustScaler()



mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)
test_csv= mms.transform(test_csv)




 


# #2. 모델구성



model = Sequential()


model = Sequential()
model.add(Dense(7, input_dim=13, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(70, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(60, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(50, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(40, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(50, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(7, activation='softmax'))

model.summary()


#3. 컴파일, 훈련

import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path1= 'C:/_data/_save/MCP/_k28/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path1, 'k28_11_', date, "_", filename]) #""공간에 ([])를 합쳐라.


x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)

#learning_rate = 1.0 
learning_rate = 0.1 
#learning_rate = 0.01 
#learning_rate = 0.001 
#learning_rate = 0.0001 


from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



es = EarlyStopping(monitor = 'val_acc', mode = 'auto', patience = 50, verbose = 1, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_acc', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)
##########################################
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience= 10,
                        mode= 'auto',
                        factor= 0.5,
                        verbose=1)
#es의 patience보다 적게 잡을것.
##########################################

model.compile(loss= 'categorical_crossentropy', optimizer= Adam(learning_rate=learning_rate), metrics= 'acc' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
start_time = time.time()
hist = model.fit(x_train, y_train, callbacks=[es, mcp, rlr], epochs= 500, batch_size = 2400, validation_split= 0.25, verbose=1)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score, r2_score
results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("정확도 :" , results[1])
print("걸린 시간 :", round(end_time - start_time, 2), "초")


loss = model.evaluate(x_test, y_test, verbose=0)
print("lr : {0}, 로스 :{1}".format(learning_rate, loss))

y_predict = model.predict(x_test, verbose=0) 
y_predict = np.argmax(y_predict,axis=1)
acc = accuracy_score(y_test, np.argmax(y_predict))

f1 = f1_score(y_test,y_predict,average='macro')
print("lr : {0}, acc :{1}".format(learning_rate, acc))

y_submit = np.argmax(model.predict(test_csv),axis=1)
y_test = np.argmax(y_test,axis=1)



f1 = f1_score(y_test,y_predict,average='macro')
print("F1: ",f1)


y_submit = le.inverse_transform(y_submit)

import datetime
dt = datetime.datetime.now()
submission_csv['대출등급'] = y_submit

submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_F1_{f1:4}.csv",index=False)



