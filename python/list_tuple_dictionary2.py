# 파이썬에서 *와 **는 어떠한 파라미터의 갯수가 정해지지 않았을때,
# 몇개의 파라티머를 받을지 모를때에 사용하게 됩니다. 
# *와 ** 둘다 함수에 다양한 수의 키워드 인수/ 매개변수를 허용하기 위해 사용합니다.


# *의 사용
# tuple의 형태로 전달됩니다. 
# arg = argument의 약자

#=== *사용
def wow(*args) :
    for a in args:
        print(a)
wow(1)
#1

wow(1,2,3)
# 1
# 2
# 3

#==============================================================

# **의 사용
# dictionary의 형태로 전달됩니다. 
# kwargs = keyword argument의 약자

#=== **사용
def kwargs(**kwargs):
    for a in kwargs:
        print(a, kwargs[a])
        
kwargs(name='abc', age=30)
# name abc
# age 30
    
 

 