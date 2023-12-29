# sessionID : 세션 ID
# userID : 사용자 ID
# TARGET : 세션에서 발생한 총 조회수
# browser : 사용된 브라우저
# OS : 사용된 기기의 운영체제
# device : 사용된 기기
# new : 첫 방문 여부 (0: 첫 방문 아님, 1: 첫 방문)
# quality : 세션의 질 (거래 성사를 기준으로 측정된 값, 범위: 1~100)
# duration : 총 세션 시간 (단위: 초)
# bounced : 이탈 여부 (0: 이탈하지 않음, 1: 이탈함)
# transaction : 세션 내에서 발생의 거래의 수
# transaction_revenue : 총 거래 수익
# continent : 세션이 발생한 대륙
# subcontinent : 세션이 발생한 하위 대륙
# country : 세션이 발생한 국가
# traffic_source : 트래픽이 발생한 소스
# traffic_medium : 트래픽 소스의 매체
# keyword : 트래픽 소스의 키워드, 일반적으로 traffic_medium이 organic, cpc인 경우에 설정
# referral_path : traffic_medium이 referral인 경우 설정되는 경로

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler


#1. 데이터
path = 'C:\\_data\\daicon\\web\\'

train_csv = pd.read_csv(path + 'train.csv', index_col=[0])
#print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=[0])
#print(test_csv)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
#print(submission_csv)

# print(train_csv.shape) #(252289, 18)
# print(test_csv.shape) #(79786, 17)
# print(submission_csv.shape) #(79786, 2)

# print(train_csv.columns) #'userID', 'TARGET', 'browser', 'OS', 'device', 'new', 'quality',
    #    'duration', 'bounced', 'transaction', 'transaction_revenue',
    #    'continent', 'subcontinent', 'country', 'traffic_source',
    #    'traffic_medium', 'keyword', 'referral_path'
    
    
#데이터 전처리    
le = LabelEncoder()
#print(test_csv.isnull().sum()) #keyword                43070/ referral_path          53891 
#print(train_csv.isnull().sum()) #keyword                137675/ referral_path          161107

#quality
#print(np.unique(train_csv['quality'], return_counts= True)) 
#print(np.unique(test_csv['quality'], return_counts= True)) 

#print(np.unique(train_csv['device'], return_counts= True)) 

x = train_csv.drop([])

# df = pd.DataFrame(x, columns= train_csv.columns)
# print(df)    
# df['Target(Y)'] = y
# print(df)    

# print('=================상관계수 히트맵==============================')    
# print(df.corr())

# print()
# #x끼리의 상관계수가 너무 높을땐 다른 데이터에 영향을 미칠수 있어(과적합확률) 컬럼처리가 필요함.
# #y와 상관관계가 있는 데이터가 중요.

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.6)
# sns.heatmap(data=df.corr(),
#             square= True,
#             annot=True,
#             cbar=True)
# plt.show()
