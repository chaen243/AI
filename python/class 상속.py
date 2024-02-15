# 클래스 상속

# 부모클래스의 속성(내용)을 자식클래스가 그대로 물려받음.
# 원래 클래스를 변경하지 않고 내부의 변수나 함수(기능)중 일부를 변경해서 사용할 때 주로 사용.

class Parents:                                    #부모클래스 정의
    def __init__(self, name='초코', age='10'):    #init으로 name, age에 각각 저장함  
        self.name = name
        self.age = age
        
        
        
class Child(Parents):                        #자식클래스 정의(상속받을 클래스이름)
     pass                                     #pass는 모든것을 상속받고 그 외 기능은 넣지 않을때 사용.
        
choco = Child()
print(choco.name)        #초코




class Child(Parents):
    def __init__(self, name='초코', age='10', category='푸들'):
        super().__init__(name, age)             #super로 상속 받을때는 'self'를 넣으면 안된다. 
        self.category = category                #super로 상속받을때는 부모클래스 외에 다른 기능 추가 가능!
        
        

choco1 = Parents()
choco2 = Child()
#print(choco1.category)      #오류 Parents class에는 category가 없기 때문
print(choco2.category)      #푸들 
print(choco1.name)      #초코
print(choco2.name)      #초코 

