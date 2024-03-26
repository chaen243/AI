import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())

tf.compat.v1.disable_eager_execution()

node1 = tf.constant(30.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
 
sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32) #variable과 거의유사.
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4})) #place holder 사용시 feed_dict가 무조건필요함. (sess.run 할때!)
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

add_and_triple = add_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4}))
#run할때의 feed_dict는 시점마다 작성해줘야함.
