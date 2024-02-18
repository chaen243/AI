import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
#print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
#print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
#print(submission_csv)

#print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())




x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)


###########################이상치처리########################


def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
    data = data.fillna(data.mean())
    return data


x = fit_outlier(x)


###########################이상치처리########################


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.85, shuffle = True, random_state=1117)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620, ) (3266, )

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

scaler = StandardScaler()
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


model = RandomForestRegressor(max_depth= 15, random_state= 42, )

#3. 훈련

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)

print("R2 스코어 :", r2)
print("loss :", loss)

#음수제거

for i in range(len(y_submit)):
    if y_submit[i] < 0:
        y_submit[i] = 123
        

submission_csv['count'] = y_submit

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
print("RMSLE: ",RMSLE(y_test,y_predict))

path = "c:\\_data\\kaggle\\bike\\"
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)


###########이전성적###########


# loss : 0.31830679029940556
# RMSE :  148.75579242627603
# RMSLE:  1.2710003152655747

###########결측치/이상치 처리 후 성적###########

# loss : 0.3486759614128119
# RMSE :  152.06986488343728
# RMSLE:  1.284835828777226


