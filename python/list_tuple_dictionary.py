#1. 리스트 (list)

fruit = ['apple', 'banana', 'mango']
print(fruit[0]) #apple                 #0부터 데이터순서가 시작됨.
fruit.append('pineapple') 
print(fruit) #['apple', 'banana', 'mango', 'pineapple']  #리스트의 가장 마지막에 데이터가 추가됨.
silence = []
print(silence) #[] 비어있는 리스트도 생성 가능.
print (len(fruit)) #4 리스트의 크기도 알아낼 수 있다.
fruit.pop(0)
print (fruit) #['banana', 'mango', 'pineapple'] 0번째 데이터 삭제.
fruit.sort() #['banana', 'mango', 'pineapple'] 순서대로 정렬해줌.
fruit.sort(reverse= True) #['pineapple', 'mango', 'banana'] 역순으로 정렬해줌.
print (fruit)

#2. 딕셔너리 (dictionary)

fruits = {'apple' : 1, 'banana' : 2, 'mango' : 3}  #딕셔너리 만들기1

#fruits = {}                  #딕셔너리 만들기2
#fruits['apple'] = 1
#fruits['banana'] = 2
#fruits['mango'] = 3



print (fruits) #{'apple': 1, 'banana': 2, 'mango': 3}
print (fruits ['apple']) #1 특정한 key의 값을 표출.

for key in fruits:
    print("key : %s \t value : %s" % (key, fruits[key]))

#데이터를 기준으로 반복작업 하는 방법
#key : apple      value : 1
#key : banana     value : 2
#key : mango      value : 3

#튜플 (Tuple)

tuple1 = ('t', 'u', 'p', 'l', 'e')
tuple2 = ( 1, 2, 3, 4, 5)
print (tuple1[1]) # u 출력. 튜플도 0부터 데이터 순서가 시작.
print (tuple1 + tuple2) # ('t', 'u', 'p', 'l', 'e', 1, 2, 3, 4, 5) 튜플끼리 연산 가능
print (tuple1 * 2) # ('t', 'u', 'p', 'l', 'e', 't', 'u', 'p', 'l', 'e')
print (len(tuple1)) # 5





