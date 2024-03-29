import numpy as np
from sklearn.preprocessing import PolynomialFeatures #데이터에 제곱한다?..


x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1] -> 0,0,1
#  [2 3] -> 4,6,9
#  [4 5] -> 16,20,25
#  [6 7]] -> 36,42,49

pf = PolynomialFeatures(degree=2, include_bias= False) 
#degree = 제곱할 값. 
#include_bias= 제곱을 0부터 할지 1부터 할지 선택하는것. 0제곱은 필요하지 않음으로 False 고정.

x_pf = pf.fit_transform(x) #원데이터 + ([0]컬럼의 제곱값, [0][1]의 곱한값, [1]의제곱값)으로 컬럼을 증폭시켜줌.
print(x_pf)
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

# pf = PolynomialFeatures(degree=3(여기까지는 잘 쓰지않음. 값이 너무 많이 늘어남), include_bias= False) 
# x_pf = pf.fit_transform(x) #원데이터 + ([0]컬럼의 제곱값, [0][1]의 곱한값, [1]의제곱값)으로 컬럼을 증폭시켜줌.
# print(x_pf)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]

x = np.arange(12).reshape(4, 3)
print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
pf = PolynomialFeatures(degree=3, include_bias= False) 
x_pf = pf.fit_transform(x) 
print(x_pf)
# [[0.000e+00 1.000e+00 2.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00
#   2.000e+00 4.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
#   0.000e+00 1.000e+00 2.000e+00 4.000e+00 8.000e+00]
#  [3.000e+00 4.000e+00 5.000e+00 9.000e+00 1.200e+01 1.500e+01 1.600e+01
#   2.000e+01 2.500e+01 2.700e+01 3.600e+01 4.500e+01 4.800e+01 6.000e+01
#   7.500e+01 6.400e+01 8.000e+01 1.000e+02 1.250e+02]
#  [6.000e+00 7.000e+00 8.000e+00 3.600e+01 4.200e+01 4.800e+01 4.900e+01
#   5.600e+01 6.400e+01 2.160e+02 2.520e+02 2.880e+02 2.940e+02 3.360e+02
#   3.840e+02 3.430e+02 3.920e+02 4.480e+02 5.120e+02]
#  [9.000e+00 1.000e+01 1.100e+01 8.100e+01 9.000e+01 9.900e+01 1.000e+02
#   1.100e+02 1.210e+02 7.290e+02 8.100e+02 8.910e+02 9.000e+02 9.900e+02
#   1.089e+03 1.000e+03 1.100e+03 1.210e+03 1.331e+03]]
