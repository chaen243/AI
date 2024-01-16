from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np

#MinMaxScaler
#데이터의 값을 최소값 0~ 최대값 1로 비례하여 변환시켜주는 Scaler입니다.
#이상치가 있는경우 변환된 값이 좁은 범위로 압축될 수 있습니다.
#데이터의 분포가 고른경우 쓰면 좋습니다.
mms= MinMaxScaler()
x = np.array([1,2,3,4,5,100]).reshape(-1,1) 
mms.fit(x)
x= mms.transform(x)
print(x)
#[[0.        ] [0.01010101] [0.02020202] [0.03030303] [0.04040404] [1.        ]]

#MaxAbsScaler
#최대절대값과 0이 각각 1, 0이 되도록 스케일링되며 즉 -1~1 사이로 변환시켜줍니다.
#양수로만 구성된 데이터셋에서는 MinMax와 비슷하게 작동하지만 큰 이상치에는 마찬가지로 값이 좁은 범위로 압축될 수 있습니다.
#데이터의 분포가 양수로만 되어있는경우, 이상치 데이터가 많이 없는경우에 사용하기 좋습니다.
mas = MaxAbsScaler()
x = np.array([0,2,3,6,10,100000]).reshape(-1,1)
mas.fit(x)
x = mas.transform(x)
print(x) 
#[[0.e+00] [2.e-05] [3.e-05] [6.e-05] [1.e-04] [1.e+00]]


x = np.array([-10000,0,2,3,6,10]).reshape(-1,1)
mas.fit(x)
x = mas.transform(x)
print(x) 
#[[-1.e+00] [ 0.e+00] [ 2.e-04] [ 3.e-04] [ 6.e-04] [ 1.e-03]]



#StandardScaler
#평균과 표준편차를 이용하여 균형잡힌 데이터셋으로 조정해줍니다.
#이상치가 있다면 평균과 표준편차에 영향을 미쳐 데이터의 확산값이 매우 달라집니다.
sds= StandardScaler()
x= np.array([1,2,3,4,5]).reshape(-1,1)
sds.fit(x)
x = sds.transform(x)
print(x)
#[[-1.41421356] [-0.70710678] [ 0.        ] [ 0.70710678] [ 1.41421356]]

sds = StandardScaler()
x= np.array([1,2,3,4,5,100000]).reshape(-1,1)
sds.fit(x)
x = sds.transform(x)
print(x)
#[[-0.44726726] [-0.44724043] [-0.4472136 ] [-0.44718676] [-0.44715993] [ 2.23606798]]



#RobustScaler
#이상치의 영향이 최소화되는 스케일링 방법입니다.
#중앙값을 제거하고 IQR에 따라 데이터의 크기를 조절합니다.
#중앙값 -25분위수와 +75분위수 사이의 범위로 조절해줍니다.
#standardScaler와 비교해보면 데이터 값을 변환해준 뒤 동일한 값을 더 넓게 분포시켜줍니다.

rds= RobustScaler()
x= np.array([1,2,3,4,5]).reshape(-1,1)
rds.fit(x)
x = rds.transform(x)
print(x)
#[[-1. ] [-0.5] [ 0. ] [ 0.5] [ 1. ]]

rds = RobustScaler()
x= np.array([1,2,3,4,5,100000]).reshape(-1,1)
rds.fit(x)
x = rds.transform(x)
print(x)
#[[-1.00000e+00] [-6.00000e-01] [-2.00000e-01] [ 2.00000e-01] [ 6.00000e-01] [ 3.99986e+04]]
