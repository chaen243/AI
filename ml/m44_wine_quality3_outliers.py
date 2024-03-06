import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


#1. 데이터 
path = "C:\\_data\\daicon\\wine\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)

test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

submit_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submit_csv.shape) #(1000, 2)

print(train_csv.columns)
#Index(['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    #   dtype='object')
    
    
    
#========라벨==========    
x = train_csv.drop(['quality'], axis=1) 
y = train_csv['quality']-3

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0
    
 
def fit_outlier(x):  
    x = pd.DataFrame(x)
    for label in x:
        series = x[label]
        q1 = series.quantile(0.05)      
        q3 = series.quantile(0.95)
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

x = fit_outlier(x)    
    
# outliers_loc = outliers(x)
# print("이상치의 위치 :", outliers_loc)    
# print("이상치의 위치 :", outliers_loc)    



# ['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
#     #    'residual sugar', 'chlorides', 'free sulfur dioxide',
#     #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#     #    'type'],

import matplotlib.pyplot as plt
plt.boxplot(x)
plt.show()

print(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=67, shuffle=True, stratify= y) #67


scaler = StandardScaler()
#scaler = MinMaxScaler()
 
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


#model = RandomForestClassifier()





#3.훈련

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)
y_predict = model.predict(x_test)

#print(y_predict)
print(y_predict.shape, y_test.shape) #(1650, 2) (1650, 2)

y_submit = model.predict(test_csv)

import time

y_submit = (y_submit)+3
#print(y_test, y_predict)
result = accuracy_score(y_test, y_predict)
submit_csv['quality'] = y_submit
acc = accuracy_score(y_predict, y_test)
ltm = time.localtime(time.time())
print("acc :", acc)
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submit_csv.to_csv(path+f"submission_{save_time}e.csv", index=False)





 
   

