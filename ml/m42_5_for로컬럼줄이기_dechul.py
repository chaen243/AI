#https://dacon.io/competitions/official/236214/data

#문자 수치화
#값 일부 자르기 (label encoder)
#값 자른거 수치화까지!

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping
import sklearn as sk
import time
import warnings
warnings.filterwarnings(action='ignore')


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
        

test_loan_perpose = test_loan_perpose.fillna(method='ffill')
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




     

from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델
Params = {'n_estimators' : 1500,
          'learning_rate' : 0.05,
          'max_depth' : 10,
          'gamma' : 0,
          'min_child_weight' : 0,
          'subsample' : 0.4,
          'colsample_bytree' : 0.8,
          'colsample_bynode' : 1,
          'reg_alpha' : 0,
          'random_state' : 1234,
          }


model = XGBClassifier(tree_method = 'hist', device = "cuda")
model.set_params( **Params)



#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score
results = model.score(x_test, y_test)
print('score :', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_predict, y_test)
print("accuracy_score :", acc)


print("걸린시간 :", round(end_time - start_time, 2), "초")




for _ in range(len(x_train[0])):
    
    feature_importances = model.feature_importances_
    
    # 가장 중요도가 낮은 피처의 인덱스 제거
    least_important_feature_index = np.argmin(feature_importances)
    x_train = np.delete(x_train, least_important_feature_index, axis=1)
    x_test = np.delete(x_test, least_important_feature_index, axis=1)
    
    #모델 재학습 및 성능 평가
    if x_train.shape[1] > 0:
        model.fit(x_train, y_train,
                  eval_set = [(x_train, y_train), (x_test, y_test)],
                  verbose = 0,
                  eval_metric = 'merror')
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(f"남은피처 갯수: {x_train.shape[1]}, {acc}")
    else:
        print("No features left,")
        break    
    
    
# score : 0.8502518303131004
# accuracy_score : 0.8502518303131004
# 남은피처 갯수: 12, 0.8505633729684823
# 남은피처 갯수: 11, 0.851238382055143
# 남은피처 갯수: 10, 0.8486941170361909
# 남은피처 갯수: 9, 0.8506152967443793
# 남은피처 갯수: 8, 0.8519653149177008
# 남은피처 갯수: 7, 0.854353808608962
# 남은피처 갯수: 6, 0.8599096526299392
# 남은피처 갯수: 5, 0.8656731917545044
# 남은피처 갯수: 4, 0.8859234643543279
# 남은피처 갯수: 3, 0.8507710680720701
# 남은피처 갯수: 2, 0.4486214237499351
# 남은피처 갯수: 1, 0.34565657614621736        