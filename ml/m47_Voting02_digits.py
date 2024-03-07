import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


datasets = load_digits() 
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify= y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델

xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()


model = VotingClassifier(
    estimators=[('LR', lr),('RF', rf),('XGB', xgb)],
    # voting= 'soft',
    # voting = 'hard' # 디폴트!
)

#3. 훈련
model.fit(x_train, y_train), 


#3. 훈련


#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

##########################
# bootstrap = True
# 최종점수 0.9666666666666667
# acc_score : 0.9666666666666667

# bootsrap = False
# 최종점수 0.9694444444444444
# acc_score : 0.9694444444444444

# soft
# acc_score : 0.9805555555555555    

#hard
# acc_score : 0.9805555555555555
    
