from keras.preprocessing.text import Tokenizer
import os
import random as rn
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler, Normalizer, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split,GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
le = LabelEncoder()
import datetime
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

SEED1 = 42
SEED2 = 42
SEED3 = 42

tf.random.set_seed(SEED1)  
np.random.seed(SEED2)
rn.seed(SEED3)

start_time = time.time()


#1. 데이터

path = 'C:\\_data\\kaggle\\Obesity_Risk\\'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#print(train_csv)
#print(test_csv)
#print(submission_csv)

# print(train_csv.shape) #(20758, 17)
# print(test_csv.shape) #(13840, 16)
# print(submission_csv.shape) #(13840, 2)

#print(train_csv.columns) #Index(['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#       'CALC', 'MTRANS', 'NObeyesdad'],



#print(train_csv['NObeyesdad'].value_counts())
# Obesity_Type_III       4046
# Obesity_Type_II        3248
# Normal_Weight          3082
# Obesity_Type_I         2910
# Insufficient_Weight    2523
# Overweight_Level_II    2522
# Overweight_Level_I     2427


##############데이터 전처리###############
le = LabelEncoder()



train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
#train_csv['WIR'] = train_csv['Weight'] / train_csv['CH2O']
train_csv['bmioncp'] = train_csv['BMI'] / train_csv['NCP']


test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
#test_csv['WIR'] = test_csv['Weight'] / test_csv['CH2O']
test_csv['bmioncp'] = test_csv['BMI'] / test_csv['NCP']

#Gender
train_csv['Gender']= train_csv['Gender'].str.replace("Male","0")
train_csv['Gender']= train_csv['Gender'].str.replace("Female","1")
test_csv['Gender']= test_csv['Gender'].str.replace("Male","0")
test_csv['Gender']= test_csv['Gender'].str.replace("Female","1")

# print(train_csv['Gender'])
# print(test_csv['Gender'])



#family_history_with_overweight
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("yes","0")
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("no","1")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("yes","0")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("no","1")

# print(train_csv['family_history_with_overweight'])
# print(test_csv['family_history_with_overweight'])

train_csv['FAVC']= train_csv['FAVC'].str.replace("yes","0")
train_csv['FAVC']= train_csv['FAVC'].str.replace("no","1")
test_csv['FAVC']= test_csv['FAVC'].str.replace("yes","0")
test_csv['FAVC']= test_csv['FAVC'].str.replace("no","1")

#print(train_csv['FAVC'])
#print(test_csv['FAVC'])
#print(np.unique(train_csv['FAVC'], return_counts= True))
#print(np.unique(test_csv['FAVC'], return_counts= True))


#print(np.unique(train_csv['CAEC'], return_counts= True))
train_csv['CAEC']= train_csv['CAEC'].str.replace("Always","0")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Frequently","1")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Sometimes","2")
train_csv['CAEC']= train_csv['CAEC'].str.replace("no","3")

test_csv['CAEC']= test_csv['CAEC'].str.replace("Always","0")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Frequently","1")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Sometimes","2")
test_csv['CAEC']= test_csv['CAEC'].str.replace("no","3")
#print(np.unique(train_csv['CAEC'], return_counts= True))
#print(np.unique(test_csv['CAEC'], return_counts= True))


#print(np.unique(test_csv['SMOKE'], return_counts= True))
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("yes","0")
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("no","1")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("yes","0")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("no","1")

#print(np.unique(train_csv['SMOKE'], return_counts= True))
#print(np.unique(test_csv['SMOKE'], return_counts= True))

#print(np.unique(train_csv['SCC'], return_counts= True))
train_csv['SCC']= train_csv['SCC'].str.replace("yes","0")
train_csv['SCC']= train_csv['SCC'].str.replace("no","1")
test_csv['SCC']= test_csv['SCC'].str.replace("yes","0")
test_csv['SCC']= test_csv['SCC'].str.replace("no","1")
#print(np.unique(test_csv['SCC'], return_counts= True))


#print(np.unique(test_csv['CALC'], return_counts= True))
test_csv['CALC']= test_csv['CALC'].str.replace("Always","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Frequently","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Sometimes","2")
test_csv['CALC']= test_csv['CALC'].str.replace("no","3")

#print(np.unique(train_csv['CALC'], return_counts= True))
train_csv['CALC']= train_csv['CALC'].str.replace("Always","0")
train_csv['CALC']= train_csv['CALC'].str.replace("Frequently","1")
train_csv['CALC']= train_csv['CALC'].str.replace("Sometimes","2")
train_csv['CALC']= train_csv['CALC'].str.replace("no","3")
#print(np.unique(train_csv['CALC'], return_counts= True))


#print(np.unique(train_csv['MTRANS'], return_counts= True))
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Automobile","0")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Bike","1")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Motorbike","2")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Public_Transportation","3")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Walking","4")

test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Automobile","0")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Bike","1")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Motorbike","2")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Public_Transportation","3")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Walking","4")



#print(np.unique(train_csv['MTRANS'], return_counts= True))
#print(np.unique(test_csv['MTRANS'], return_counts= True))


#print(test_csv.isnull().sum()) #없음.
#print(train_csv.isnull().sum()) #없음.

# (['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad', 'BMI'],


x = train_csv.drop(['NObeyesdad'], axis = 1)

def fit_outlier(x):  
    x = pd.DataFrame(x)
    for label in x:
        series = x[label]
        q1 = series.quantile(0.20)      
        q3 = series.quantile(0.80)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        x[label] = series
        
    x = x.fillna(x.mean())
    return x

y = train_csv['NObeyesdad']
y = le.fit_transform(y)
#test_csv = test_csv.drop(['FAVC'], axis = 1)



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
import time


mms = MinMaxScaler()
# mms = MinMaxScaler(feature_range=(0,1))
# mms = StandardScaler()
# mms = MaxAbsScaler()
# mms = RobustScaler()

mms.fit(x)
x = mms.transform(x)
test_csv=mms.transform(test_csv)
#print(x.shape, y.shape)  #(20758, 16) (20758,)
#print(np.unique(y, return_counts= True))
#(array([0, 1, 2, 3, 4, 5, 6]), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))


# df = pd.DataFrame(x, columns= train_csv.columns)
# print(df)    
# df['Target(Y)'] = y
# print(df)    

# print('=================상관계수 히트맵==============================')    
# print(df.corr())

#x끼리의 상관계수가 너무 높을땐 다른 데이터에 영향을 미칠수 있어(과적합확률) 컬럼처리가 필요함.
#y와 상관관계가 있는 데이터가 중요.

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.6)
# sns.heatmap(data=df.corr(),
#             square= True,
#             annot=True,
#             cbar=True)
# plt.show()








x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=SEED1, stratify= y)
#5 9158 19 9145

#mms = MinMaxScaler() #(feature_range=(0,1))
mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()
#mms = Normalizer()


mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)
test_csv= mms.transform(test_csv)


#==================================
#2. 모델
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA #차원축소


################lgbm########################
params = {
    'n_estimators': 4000,
    'learning_rate': 0.005 ,
    'reg_alpha': 0.7,
    "verbosity": 1,  
    'reg_lambda': 0.9935667255875388,
    'max_depth': 12,
    'num_leaves': 30,
    'min_child_weight': 1, #4
    'subsample': 0.393274050086088,
    'colsample_bytree': 0.4579828557036317,
    'n_jobs' : -1
}



model = XGBClassifier(**params)


model.set_params(learning_rate=0.003,
                 n_estimators=4500,
                 max_depth=18, 
                 reg_alpha=0, 
                 min_child_weight = 4,
                 early_stopping_rounds= 50
                 )


################lgbm########################




#3. 컴파일 , 훈련

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)



start_time = time.time()
model.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_test, y_test)],
          verbose = 1)
end_time = time.time()




#4. 평가, 예측


results = model.score(x_test, y_test)
print('score :', model.score(x_test, y_test))

print("점수1 :", results)


y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
y_submit = le.inverse_transform(y_submit)



#########catboost###########
#y_submit = model.predict(test_csv)[:,0]



acc = model.score(x_test, y_test)
end_time = time.time()





results2 = model.score(x_test, y_test)
print('최종점수2 : ',results2)

print('걸린시간 :', round(end_time - start_time,2),'초')
import datetime
dt = datetime.datetime.now()
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{acc:4}.csv",index=False)


import pickle

path = "C:\\_data\\_save\\_kaggle_Obesity_Risk\\"
pickle.dump(model, open(path + f'submit_{dt.day}day{dt.hour:2}{dt.minute:2}_acc_{acc:4}.dat', 'wb'))

#이전 성적

# score : 0.9205202312138728
# acc : 0.9205202312138728
# 0.9205202312138728

#이상치.결측치 처리 후 성적

# score : 0.9210019267822736
# acc : 0.9210019267822736
# 0.9210019267822736