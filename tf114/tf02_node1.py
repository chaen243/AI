import tensorflow as tf

# 3 + 4
node1 = tf.constant(3.0, tf.float32) # 3.0이 들어가고 float32 형태의 데이터다 까지 정의해야함 (명시하지 않으면 자동으로 32로 들어감.)
node2 = tf.constant(4.0, tf.float32) # 4.0이 들어가고 float32 형태의 데이터다 까지 정의해야함
node3 = tf.add(node1, node2) 

print(node3) #Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session() #실행시킬 변수!
print(sess.run(node3)) #최종연산만 run시키면 됨
# node3을 연산 시켜라! (위의 과정까지 모두 실행 시키는 방식.)

