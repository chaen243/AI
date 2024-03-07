import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from catboost import CatBoostClassifier
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

model = StackingClassifier(
    estimators=[('LR',lr),('RF',rf),('XGB',xgb)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs= -1,
    cv=5, 
)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('스태킹 acc :', accuracy_score(y_test, y_pred))


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

#stacking
# model.score : 0.9833333333333333
    
