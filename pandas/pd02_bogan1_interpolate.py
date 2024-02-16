#interpolate 결측치 처리, 이상치 처리에 주로 사용!!
'''
결측치 처리
1. 행 또는 열 삭제
2. 임의의 값 채우기 (fillna -0, ffill(앞의 값), bfill(뒤의 값), 중위값(median), 평균값(mean), 특정값(조건을 달아서 넣을때)....)
3. 보간 (interpolate)
4. 모델을 만들어서 predict 함
5. 부스팅계열 : 결측치 이상치에 대해 자유롭다. (결측치 처리 안해도 돌아감) 개쩐다..(과적합 위험은 있긴해서 결측치 처리는 하는게 좋음)
같은 모델로 돌리게 되면 같은 값이 나올 확률이 높다.


'''
import pandas as pd
from datetime import datetime
import numpy as np

dates = ['2/16/2024', '2/17/2024', '2/18/2024',
         '2/19/2024', '2/20/2024', '2/21/2024',]
dates = pd.to_datetime(dates) #시계열. 날짜종류로 변형할때 써줌
print(dates)

print('='*100)
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index = dates)

print(ts)
# 2024-02-16     2.0
# 2024-02-17     NaN
# 2024-02-18     NaN
# 2024-02-19     8.0
# 2024-02-20    10.0
# 2024-02-21     NaN
print('='*100)
ts = ts.interpolate() 
print(ts)