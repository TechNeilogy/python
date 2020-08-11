import tensorflow as tf

A = tf.constant([[0, 1], [2, 3]])

B = tf.matrix_transpose(A)

C = tf.matmul(A, B)

D = tf.matmul(B, A)

E = C + D

with tf.Session() as sess:
    print(sess.run(E))