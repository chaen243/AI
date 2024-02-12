
# - try-except문
# try블록 수행 중 오류가 발생하면 except 블록이 수행되고 오류가 발생하지 않으면 except 블록은 수행되지 않습니다.

# except구문은 
# 1. try:
#    except:   로 오류의 종류와 상관없이 오류가 발생했을때 예외처리를 할 수 있습니다.
# 2. try:
#    except 발생오류:   로 미리 정해 놓은 오류와 같은 오류가 발생했을때만 예외처리를 수행 할 수 있습니다.
# 3. try:
#    except 발생오류 as 오류변수:   로 오류의 내용을 알고 싶을때 사용 할 수 있습니다.




# - try-finally 문
# finally절은 예외 발생 여부에 상관 없이 항상 수행됩니다. 중간에 오류가 발생 하더라도 무조건 실행됩니다.


# -try-else 문
# 오류가 발생하면 except, 오류가 발생하지 않으면 else절을 수행되게 됩니다.


# -pass
# 오류가 발생했을때 pass를 사용하여 오류를 회피하게 할 수 있습니다.


# -raise
# 오류를 강제로 발생 시킬 수 있습니다.


# -예외 만들기
# 파이썬 내장클래스인 Exception을 사용해 만들 수 있습니다.



try:
    f = open("파일", 'r') 
except FileNotFoundError: #발생오류 넣기
    pass
print('done')    
#done

try:
    f = open("파일", 'r') 
except : #오류종류 상관없이 예외처리
    pass
print('done')    
#done



#예외 만들기
class MyError(Exception):
    pass

def say_hi(hi):
    if hi == '안녕':   
        raise MyError() # raise=오류를 강제로 실행시키는 명령어
    print(hi) 
    
try:
    say_hi('안녕')
    say_hi('바보')
except MyError:
    print('stupid')       
    
#stupid   
       
































