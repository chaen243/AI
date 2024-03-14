import tensorflow as tf
print(tf.__version__) #1.14.0

#print('helloworld')

'''
hello = tf.constant('hello world')
print(hello) # constant 상태로 준비되이있음
'''

hello = tf.constant('hello world')
print(hello) # constant 상태로 준비되이있음

sess = tf.Session() #세선 정의
print(sess.run(hello)) #b'hello world'