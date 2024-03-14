import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([3], dtype=tf.float32)

#tensor1에서는 변수 초기화를 항상 꼭 해줘야함.
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(a + b))