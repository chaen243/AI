import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(2.0)

#실습내용
# 덧셈 : node3
# 뺄셈 node4
# 곱셈 5
# 나눗셈 6

node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)



sess = tf.Session()
print(sess.run(node3)) #4
print(sess.run(node4)) #0 
print(sess.run(node5)) #4
print(sess.run(node6)) #1
