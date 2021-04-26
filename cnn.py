import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


# Repair dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)


# Create a custom regularization function
class L2Regularization(keras.regularizers.Regularizer):
    def __init__(self, weight):
        super(L2Regularization, self).__init__()
        self.weight = weight

    def __call__(self, x):

        return self.weight * tf.reduce_sum(tf.square(x))

    def get_config(self):

        return {'weight': self.weight}


# Model
def cnn_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.02))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=L2Regularization(weight=0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.Dropout(rate=0.5)(x)
    outputs = layers.Dense(units=10)(x)
    model = keras.Model(inputs, outputs)

    return model


# Loss_fn
model = cnn_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics='accuracy'
)

# Train and evaluate
model.fit(x_train, y_train, batch_size=16, epochs=3, verbose=2)
model.evaluate(x_test, y_test)
