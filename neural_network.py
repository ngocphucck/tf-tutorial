import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# Train, test loader
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalization
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255
# Convert to tensor
x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)


# Model (Using sequential API)
model = keras.Sequential(
    [
        layers.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(10)
    ]
)

# model = keras.Sequential()
# model.add(layers.Input(28*28))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(10, activation='relu'))
#

# # Model (Functional API)
# inputs = layers.Input(28 * 28)
# x = layers.Dense(512, activation='relu')(inputs)
# outputs = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluation
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# Prediction
classes = model.predict(x_test[:3])
print(classes)
