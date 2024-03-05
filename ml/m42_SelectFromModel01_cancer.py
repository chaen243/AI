import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

Params = {'n_estimators' : 1000,
          'learning_rate' : 0.01,
          'max_depth' : 3,
          'gamma' : 0,
          'min_child_weight' : 0,
          'subsample' : 0.4,
          'colsample_bytree' : 0.8,
          'colsample_lylevel' : 0.7,
          'colsample_bynode' : 1,
          'reg_alpha' : 0,
          'reg_lamba' : 1,
          'random_state' : 3377,
          'verbose' : 0        
          }

#2. 모델


model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **Params)

#3. 훈련
model.fit(x_train, y_train, 
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 1,
          eval_metric = 'logloss')

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)

##########################

threadsholds = np.sort(model.feature_importances_)  
#넘파이 형태로 들어가있음
print(threadsholds)
print('='*100)
# [0.00872861 0.00893266 0.0093171  0.01000617 0.01032185 0.01137479
#  0.01148892 0.01181221 0.01186208 0.01308961 0.01325941 0.01502781
#  0.01733723 0.0176222  0.01830023 0.01864554 0.01885716 0.01909443
#  0.01958342 0.0200609  0.02312445 0.0257998  0.03260686 0.03543241
#  0.0373496  0.05666851 0.08028697 0.11175235 0.14107467 0.1711819 ]

from sklearn.feature_selection import SelectFromModel
# 중요도가 작은 것부터 순차적으로 컬럼의 갯수를 줄여줌


for i in threadsholds:
    # i보다 크거나 같은것만 남겨놓음
    selection = SelectFromModel(model, threshold=i, prefit=False)
    #클래스를 인스턴스 (변수)화 한다
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test) 
    #print(i,  "\t변형된x_train :", select_x_train.shape, '변형된x_test :', select_x_test.shape)
    
    select_model = XGBClassifier()
    select_model.set_params(
        early_stopping_rounds = 10,
        **Params,
        eval_metric= 'logloss'
    ) 
    
    select_model.fit(select_x_train, y_train,
                     eval_set= [(select_x_train, y_train), (select_x_test, y_test)],verbose=0)
    
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print("Thredsholds=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))
    
    