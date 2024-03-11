import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

#1. 데이터

x,y = load_linnerud(return_X_y=True)
print(x.shape) #(20,3)
print(y.shape) #(20,3)
# 최종값 : x [  2. 110.  43.]] y [138.  33.  68.]]

#2. 모델
model = RandomForestRegressor()
#model = Ridge()
# model = Lasso()
# model = XGBRegressor()
#model = LinearRegression()
#model = LGBMRegressor() 에러 (multioutput xx)
#model = CatBoostRegressor() 에러 (multioutput xx)

#3. 훈련
model.fit(x, y)

#4. 평가
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
      round(mean_absolute_error(y, y_pred), 4))
print(model. predict([[2, 110, 43]]))

# Ridge 스코어 : 7.4569
# LinearRegression 스코어 : 7.4567
# Lasso 스코어 : 7.4629
# XGBRegressor 스코어 : 0.0008
# RandomForestRegressor 스코어 : 3.6827


# model = MultiOutputRegressor(LGBMRegressor())
# MultiOutputRegressor 스코어 : 8.91
#param_set 여기서 사용하면 multioutputregressor의 파라미터가 되기때문에
#LGBM파라미터 사용하려면 **param으로 만들어 사용


#catboost는 방법2가지
# model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
# MultiOutputRegressor 스코어 : 0.2154
model = CatBoostRegressor(loss_function='MultiRMSE')
# CatBoostRegressor 스코어 : 0.0638

#3. 훈련
model.fit(x, y)

#4. 평가
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
      round(mean_absolute_error(y, y_pred), 4))
print(model. predict([[2, 110, 43]]))
