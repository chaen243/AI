import pandas as pd

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
    
]

index = ["031", "059", "033","045", "023"]
columns = ["종목명", "시가", "종가"]

df= pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

print("="*100)
#print(df[0]) #에러
#print(df["031"]) #에러
print(df["종목명"]) #가능...? 판다스의 기준은 컬럼(열)! 컬럼명으로 꼭 넣을것.
#아모레를 출력하고싶을때
# print(df[4,0]) key에러
# print(df['종목명','045']) key에러
print(df['종목명']['045']) #아모레
####판다스는 열,행!!!

'''
#loc : 인덱스를 기준으로 행 데이터 추출
#iloc : 행번호를 기준으로 행 데이터 추출
    # 인트(숫자 넘버)loc
'''    
    
print("="*100)
#print(df.loc[3]) # 키에러
print(df.loc['045']) 
# 종목명     아모레
# 시가     3500
# 종가     6000

print("="*100)

print(df.iloc[3])
# 종목명     아모레
# 시가     3500
# 종가     6000

print("="*20,'네이버',"="*20)
print(df.loc['023'])
# 종목명     네이버
# 시가      100
# 종가     1500
print(df.iloc[4])
# 종목명     네이버
# 시가      100
# 종가     1500
print(df.iloc[-1])
# 종목명     네이버
# 시가      100
# 종가     1500

print("="*20,'아모레시가',"="*20)
print(df.loc['045'].loc['시가'])
print(df.loc['045'].iloc[1])
print(df.iloc[3,1])
print(df.iloc[3].iloc[1])
print(df.iloc[3].loc['시가'])
print(df.loc['045']['시가']) 
print(df.iloc[3]['시가']) 
print(df.loc["045", '시가'])


#워닝뜨지만 사용가능
print(df.loc['045'][1]) 
print(df.iloc[3][1])

print('=========아모레, 네이버 시가=======')
print(df.iloc[3:5,1])
print(df.iloc[[3,4],1])
#print(df.iloc[3:5,"시가"]) #에러
#print(df.iloc[[3,4],"시가"]) #에러
#print(df.loc[['045','023'],1]) #에러
print(df.loc[['045','023'],'시가']) #가능
print(df.loc['045':'023','시가']) #가능

