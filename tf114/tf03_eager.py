import tensorflow as tf
print(tf.executing_eagerly()) #False

#즉시실행모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행시킴.
#즉시실행모드 켬

#tf.compat.v1.disable_eager_execution() #즉시실행모드 끔./ 텐서1 문법/ 디폴트
tf.compat.v1.enable_eager_execution() #False #즉시실행켬. 텐서2 사용가능

print(tf.executing_eagerly())

hello = tf.constant('Hello World')

sess = tf.Session()
print(sess.run(hello))

#가상환경   즉시실행          사용가능여부
#1.14.0    disable (디폴트)     가능!!!
#1.14.0    enable               에러
# 2.9.0    disable              가능!!!  
# 2.9.0    enable (디폴트)      에러
