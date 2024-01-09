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

'''
공통적으로는 모두 여러개의 데이터를 담을 수 있는 구조.

리스트- [대괄호]를 이용해 만들고, 데이터는 각각 ','를 사용해 구분합니다.
순서가 있어 인덱싱이 가능합니다.
여러개의 값을 넣을 수 있고 리스트는 0번째가 첫번째 데이터입니다.
데이터 생성과 삭제가 자유롭습니다.
리스트 안에 리스트를 넣을수도 있고 비어있는 리스트도 만들 수 있습니다.
List.append(x) 함수를 통해 리스트의 마지막에  x값을 추가해줄수 있습니다.

딕셔너리-{중괄호}를 이용해 만들고 {key:value, key:value의 형태로 만듭니다.
순서가 없어 인덱싱이 불가능합니다.
key 데이터는 중복과 변경이 되지 않습니다.
여러가지 값을 저장해놓고 필요한 값을 꺼내어 쓸 수 있습니다.
필요에 따라 가져올 값의 key를 알고 있으면 바로 접근이 가능합니다.


튜플- (소괄호)를 이용하여 만들고 한번 만들어진 후에는 데이터들을 변경 혹은 삭제 할 수 없습니다. 순서는 존재하지만 변경이 불가능합니다. 알고리즘으로 변경되면 안되는 값이 변경되는 것을 방지할 수 있습니다.




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
'''




