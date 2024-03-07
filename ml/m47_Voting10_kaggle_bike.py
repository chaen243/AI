import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVR


#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())


x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#2. 모델구성
xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LinearSVR()
 
model = VotingRegressor(
    estimators=[('LR', lr),('RF', rf),('XGB', xgb)],
    # voting= 'soft',
    # voting = 'hard' # 디폴트!
)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()



print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))


print("걸린시간 :", round(end_time - start_time, 2), "초")




#true
# r2_score : 0.1885203279242249

#false
# score : 0.18713394885890466

#voting
# score : 0.3335607416963229