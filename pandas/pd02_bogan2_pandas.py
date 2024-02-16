import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan,8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
#print(data)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#결측치 확인
print(data.isnull()) #True가 결측치
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
print(data.info())
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

#결측치 삭제

#print(data.dropna()) #= print(data.dropna(axis=0))
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

#print(data.dropna(axis=1))
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

#2-1. 특정값 - 평균

means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 = 중위값

median = data.median()
print(median) 
data3 = data.fillna(median)
print(data3)

#2-3. - 0채우기

data4 = data.fillna(0)
print(data4)


data41= data.fillna(777) #채울값만 입력하면 됨.
print(data41)

#2-4. 특정값 ffill #시계열 데이터에서 많이 사용

#data5 = data.fillna(method='ffill' )
data5 = data.ffill() #똑같음!

print(data5)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

#2-5. 특정값 bfill

data6 = data.fillna(method='bfill' ) #시계열 데이터에서 많이 사용
data6 = data.bfill() #똑같음!

print(data6)

######특정컬럼만 채우기##############
#컬럼 지정해서 위와 같이 하면 됨!
mean = data['x1'].mean()
print(mean)                 #6.5

med = data['x4'].median()
print(med)              #6.0

data['x1'] = data['x1'].fillna(mean)
data['x4'] = data['x4'].fillna(med)
data['x2'] = data['x2'].ffill()

print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0