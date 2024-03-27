import tensorflow as tf
tf.compat.v1.set_random_seed(777)

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')

#초기화첫번째
sess= tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa=sess.run(변수)
print('aaa', aaa) #aaa [ 2.2086694  -0.73225045]
sess.close()

#초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session = sess) #텐서플로의 데이터형인 '변수'를 파이썬에서 볼 수 있게 바꿔줌.
print('bbb', bbb) #ccc [ 2.2086694  -0.73225045]
sess.close()
#마찬가지로 닫아줘야함.

#초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc= 변수.eval() #제일편함..!
print('ccc',ccc) #ccc [ 2.2086694  -0.73225045]
sess.close()


