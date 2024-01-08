#클래스의 예
class User:                          #클래스는 class로 정의  클래스 안에는 여러가지 함수를 넣을 수 있음
   def __init__(self, name, age) :   #init은 자동으로 함수를 실행시켜줌
       self.name = name              #함수는 def로 정의.
       self.age = age
          
   def hello(self):                  #self는 클래스를 정의할때의 약속과도 같음. 
       print(f"{self.name, self.age}")                 #사용되지는 않으나 꼭 기입하여야 클래스가 작동함    
    
User_1 = User("피카츄", 20)
User_2 = User("라이츄", 30)
User_3 = User("파이리", 25)

print(User_1.name)
print(User_1.age)

User_1.hello()
User_2.hello()
User_3.hello()