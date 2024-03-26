import tensorflow as tf
print(tf.__version__) #1.14.0

#print('helloworld') tf2.x에서만 가능

'''
hello = tf.constant('hello world')
print(hello) # constant 상태로 준비되이있음
'''

hello = tf.constant('hello world')
print(hello) # constant 상태로 준비되이있음 (변하지 않는 상태.)

sess = tf.Session() #세선 정의 (객체로 만들어줌)
print(sess.run(hello)) #b'hello world'