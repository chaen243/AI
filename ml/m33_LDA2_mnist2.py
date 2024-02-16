#m31_1에서 뽑은 4가지 결과로 4가지 모델 만들기!
#시간.성능 체크
#결과 예시
# 결과1 PCA=154
# 걸린시간 000초
# acc= 0.000

#xgboost, 그리드, 랜덤, halving중 사용
#m31-2보다 성능 좋게 만들기
# parameters = [{
#     'n_estimators': [4000, 3000],
#     'learning_rate': [0.92, 0.62],
#     'gamma': [0.8024196, 0.785872],
#     'reg_alpha': [0.99025931173755949, 0.9469289],
#     'reg_lambda': [0.8835667255875388],
#     'max_depth': [40, 43],
#     'min_child_weight': [1,4],
#     'subsample': [0.393274050086088],
#     'colsample_bytree': [0.4579828557036317]
# }]


# tree_method= 'gpu_hist',
# predictor='gpu_predictor',
# gru_id=0


#scaler
# print(np.argmax(evr_cunsum >= 1.0) +1) #1
# print(np.argmax(evr_cunsum >= 0.999) +1) #683
# print(np.argmax(evr_cunsum >= 0.99) +1) #544
# print(np.argmax(evr_cunsum >= 0.95) +1) #332
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split,GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV, StratifiedKFold
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


(x_train, y_train), (x_test,y_test) = mnist.load_data()# '_'자리에는 데이터를 받지 않음.
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)
#x = np.append(x_train, x_test, axis=0)
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

#print(x.shape) #(70000, 28, 28)
#print(y.shape) #(70000,)
x = x.reshape(70000, 28*28) #(70000, 784)
#y = pd.get_dummies(y)

#print(x.shape) # #(70000, 784)

scaler = StandardScaler()
x_1 = scaler.fit_transform(x)
lda = LinearDiscriminantAnalysis()
x_1 = lda.fit_transform(x,y)  
y = pd.get_dummies(y)


x_train, x_test, y_train, y_test = train_test_split(x_1, y, shuffle=True, random_state=123, stratify= y)
 
model = XGBClassifier(tree_method= 'hist', 
                           device= 'cuda', 
                           n_estimators= 500,
                           learning_rate = 0.38,
                           max_depth= 8,
                           min_child_weight= 3,
                           colsample_bytree= 0.8,
                          n_jobs=-5)

    #3. 컴파일, 훈련
start_time = time.time()
model.fit( x_train, y_train)#, batch_size=500, verbose=1, epochs= 200, validation_split=0.2, )
end_time = time.time()

    #4. 평가, 예측
results = model.score(x_test,y_test)
print('=========================')
print('acc:',  results)
print('걸린시간: ', round(end_time - start_time,2), '초')
    

#=============================

# acc: 0.9804571270942688
# PCA 683
# 걸린시간:  14.72 초


# acc: 0.9799428582191467
# PCA 544
# 걸린시간:  27.81 초



# acc: 0.9801714420318604
# PCA 332
# 걸린시간:  27.23 초



# acc: 0.9812571406364441
# PCA 1
# 걸린시간:  28.4 초

# acc: 0.8936571428571428
# 걸린시간:  8.49 초