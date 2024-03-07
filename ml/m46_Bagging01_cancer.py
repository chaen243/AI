#배깅 보팅  부스팅 스태킹

'''
배깅- randomforest
모델이 1종류. 사용하는 데이터가 다르다. 
트리마다 데이터를 랜덤하게 가져와 샘플링을 다르게 하여 훈련함. 랜덤한 데이터중 중복해서 가져오기 가능.
(n_estimator) 디시전트리를 한번 훈련시켰다. 에포와 비슷!



보팅- 투표 (모델 여러개중 좋은거/평균값 사용하겠다)
hard - 단순하게 값이 좋은게 많은 것으로 사용. 다수결. (주로 모델갯수를 홀수로 사용)
soft - 값을 평균내서 그중 제일 좋은 모델을 채택.


부스팅- 가중치가 낮은 값에 가중치를 더 줘서 
'''

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier # 어떤 모델이라도 배깅방식으로 사용 가능!
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


datasets = load_breast_cancer()


x = datasets.data
y= datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Params = {'n_estimators' : 1000,
#           'learning_rate' : 0.01,
#           'max_depth' : 3,
#           'gamma' : 0,
#           'min_child_weight' : 0,
#           'subsample' : 0.4,
#           'colsample_bytree' : 0.8,
#           'colsample_lylevel' : 0.7,
#           'colsample_bynode' : 1,
#           'reg_alpha' : 0,
#           'reg_lamba' : 1,
#           'random_state' : 3377,
#           'verbose' : 0        
#           }

#2. 모델
xgb = XGBClassifier()
#xgb.set_params(**Params)
#배깅 안에 모델을 래핑
model = BaggingClassifier(LogisticRegression(),
                          n_estimators= 10,
                          n_jobs= -1,
                          random_state= 777,
                          bootstrap=True, #디폴트, 중복허용! (과적합 방지!)
                        # bootstrap=False #중복 허용하지 않음xxxx
                          )

#3. 훈련
model.fit(x_train, y_train), 
          #eval_set = [(x_train, y_train), (x_test, y_test)],
          #verbose = 1,
          #eval_metric = 'logloss')

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score :', acc)


    
    
