import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


# Initialization of Tensors
x = tf.constant(4, shape=(2, 2, 3), dtype=tf.float32)
x = tf.constant([[2, 3, 1], [1, 1, 1]], dtype=tf.float64)
x = tf.ones((3, 3))
x = tf.eye(3)  # Identity matrix
x = tf.random.normal((2, 3), mean=1, stddev=0.6)
x = tf.random.uniform((4, 3), minval=0, maxval=2)

x = tf.cast(x, dtype=tf.int32)  # Casting between 2 data types

# Mathematics operations
x = tf.constant([4, 2, 3])
y = tf.constant([1, 2, 1])

z = x + y
z = x - y
z = x * y

z = tf.tensordot(x, y, axes=1)
z = x ** 2

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 5))
z = tf.matmul(x, y)

# Indexing
x = tf.constant([2, 1, 5, 6, 3, 4, 1, 4])

indices = tf.constant([0, 1, 3])
z = tf.gather(x, indices)

# Reshaping
x = tf.reshape(x, (2, 4))
x = tf.transpose(x)
print(x)
